"""
bwa_backend.py  —  Blog Writing Agent Backend
================================================
All LangGraph logic lives here.
API keys are injected at runtime via configure_keys()
so the Streamlit frontend can pass them in from the sidebar.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing import TypedDict

load_dotenv()

# ─────────────────────────────────────────────
# Global model placeholder (set by configure_keys)
# ─────────────────────────────────────────────
_model = None


def configure_keys(groq_api_key: str, tavily_api_key: str, gemini_api_key: str = "") -> None:
    """
    Call this once from the Streamlit frontend before invoking the graph.
    Sets environment variables so all libraries pick them up automatically.
    """
    global _model

    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    else:
        # Clear it so image generation skips gracefully
        os.environ.pop("GOOGLE_API_KEY", None)

    # Re-initialise the LLM with the new key
    _model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)


def _get_model():
    """Return the configured model, or raise a clear error."""
    if _model is None:
        raise RuntimeError(
            "No API keys configured. Call bwa_backend.configure_keys() first."
        )
    return _model


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g [[IMAGE-1]]")
    filename: str = Field(..., description="filename e.g attention.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt sent to image model")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_place_holder: str
    images: List[ImageSpec]


class RouterDecision(BaseModel):
    need_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str]


class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem]


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader understands after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=5)
    target_words: int = Field(..., description="Target word count 120–450.")
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ]
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class State(TypedDict):
    topic: str
    # routing / research
    mode: str
    need_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    # planning
    plan: Optional[Plan]
    # parallel section writing
    section_md: Annotated[List[tuple], operator.add]
    # reducer
    merged_md: str
    md_place_holders: str
    image_spec: List[dict]
    final: str


# ─────────────────────────────────────────────
# Node 1 — Router
# ─────────────────────────────────────────────

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
- closed_book (need_research=false): Evergreen topics where correctness does not depend on recent facts.
- hybrid (need_research=true): Mostly evergreen but needs up-to-date examples/tools/models.
- open_book (need_research=true): Mostly volatile — weekly roundups, "latest", rankings, pricing.

If need_research=true: output 3–10 high-signal, scoped queries.
"""


def router_node(state: State) -> dict:
    model = _get_model()
    decision = model.with_structured_output(RouterDecision).invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    return {
        "mode": decision.mode,
        "need_research": decision.need_research,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["need_research"] else "orchestrator"


# ─────────────────────────────────────────────
# Node 2 — Researcher
# ─────────────────────────────────────────────

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects.
Rules:
- Only include items with a non-empty url.
- Prefer authoritative sources (company blogs, docs, reputable outlets).
- Keep published_at as YYYY-MM-DD if present; else null. Do NOT guess.
- Keep snippets short. Deduplicate by URL.
"""


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized = []
    for r in results:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized


def research_node(state: State) -> dict:
    model = _get_model()
    queries = state.get("queries", [])
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    pack = model.with_structured_output(EvidencePack).invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results: {raw}"),
    ])

    dedup = {e.url: e for e in pack.evidence if e.url}
    return {"evidence": list(dedup.values())}


# ─────────────────────────────────────────────
# Node 3 — Orchestrator
# ─────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = (
    "You are a senior technical writer and developer advocate. "
    "Produce a highly actionable outline for a technical blog post.\n\n"
    "Hard requirements:\n"
    "- Create 5–7 sections (tasks).\n"
    "- Each section must include: goal, 3–5 bullets, target word count (120–450).\n"
    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n"
    "- Assume the reader is a developer; use correct terminology.\n"
    "- Bullets must be actionable (e.g. 'Show a minimal code snippet for X').\n"
    "- Include at least ONE of: MWE/code sketch, edge cases, performance/cost, "
    "security, debugging tips.\n"
    "Output must strictly match the Plan schema."
)


def orchestrator_node(state: State) -> dict:
    model = _get_model()
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = model.with_structured_output(Plan).invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
        )),
    ])
    return {"plan": plan}


# ─────────────────────────────────────────────
# Node 4 — Fanout + Worker
# ─────────────────────────────────────────────

def fanout(state: State):
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"].model_dump(),
            "mode": state["mode"],
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = (
    "You are a senior technical writer. Write ONE section of a technical blog post in Markdown.\n\n"
    "Hard constraints:\n"
    "- Follow the Goal and cover ALL Bullets in order.\n"
    "- Stay within ±15% of Target words.\n"
    "- Output ONLY the section content in Markdown (no H1 title, no commentary).\n"
    "- Start with a '## <Section Title>' heading.\n"
    "- Be precise and implementation-oriented.\n"
    "- Include code snippets, checklists, or diagrams where helpful.\n"
    "- Avoid fluff and marketing language.\n"
)


def worker_node(payload: dict) -> dict:
    model = _get_model()
    task = payload["task"]
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = model.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Section: {task.title}\n"
            f"Section type: {task.section_type}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Bullets:{bullets_text}\n"
            f"Tags: {task.tags}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Evidence (only use these URLs when citing):\n{evidence_text}\n"
        )),
    ]).content.strip()

    return {"section_md": [(task.id, section_md)]}


# ─────────────────────────────────────────────
# Reducer — subgraph
# ─────────────────────────────────────────────

def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered = [md for _, md in sorted(state["section_md"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    Path(f"{plan.blog_title}.md").write_text(merged_md, encoding="utf-8")
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.
Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    model = _get_model()
    plan = state["plan"]
    image_plan = model.with_structured_output(GlobalImagePlan).invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Topic: {state['topic']}\n"
            f"Insert placeholders + propose image prompts.\n\n"
            f"{state['merged_md']}"
        )),
    ])
    return {
        "md_place_holders": image_plan.md_place_holder,
        "image_spec": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """Generate image bytes via Gemini. Raises RuntimeError if key missing or call fails."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set — skipping image generation.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned from Gemini.")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in Gemini response.")


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_place_holders") or state["merged_md"]
    image_specs = state.get("image_spec", []) or []

    # If no Gemini key is configured — skip image generation entirely
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY", "").strip())

    if not image_specs or not has_gemini:
        Path(f"{plan.blog_title}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # Graceful fallback — keep doc usable
                fallback = (
                    f"> **[Image skipped]** {spec.get('caption', '')}\n"
                    f"> *{spec.get('alt', '')}*\n"
                    f"> Error: {e}\n"
                )
                md = md.replace(placeholder, fallback)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    Path(f"{plan.blog_title}.md").write_text(md, encoding="utf-8")
    return {"final": md}


# ─────────────────────────────────────────────
# Build reducer subgraph
# ─────────────────────────────────────────────

_reducer = StateGraph(State)
_reducer.add_node("merge_content", merge_content)
_reducer.add_node("decide_images", decide_images)
_reducer.add_node("generate_and_place_images", generate_and_place_images)
_reducer.add_edge(START, "merge_content")
_reducer.add_edge("merge_content", "decide_images")
_reducer.add_edge("decide_images", "generate_and_place_images")
_reducer.add_edge("generate_and_place_images", END)
reducer_subgraph = _reducer.compile()


# ─────────────────────────────────────────────
# Build main graph
# ─────────────────────────────────────────────

_g = StateGraph(State)
_g.add_node("router", router_node)
_g.add_node("research", research_node)
_g.add_node("orchestrator", orchestrator_node)
_g.add_node("worker", worker_node)
_g.add_node("reducer", reducer_subgraph)

_g.add_edge(START, "router")
_g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
_g.add_edge("research", "orchestrator")
_g.add_conditional_edges("orchestrator", fanout, ["worker"])
_g.add_edge("worker", "reducer")
_g.add_edge("reducer", END)

app = _g.compile()


# ─────────────────────────────────────────────
# Convenience run() for CLI testing
# ─────────────────────────────────────────────

def run(topic: str) -> dict:
    return app.invoke({
        "topic": topic,
        "mode": "",
        "need_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "section_md": [],
        "md_place_holders": "",
        "image_spec": [],
        "final": "",
    })


if __name__ == "__main__":
    # For local CLI testing — keys loaded from .env via load_dotenv() above
    configure_keys(
        groq_api_key=os.environ["GROQ_API_KEY"],
        tavily_api_key=os.environ["TAVILY_API_KEY"],
        gemini_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )
    result = run("write a short blog on self attention")
    print("\n===== FINAL =====\n")
    print(result["final"])
