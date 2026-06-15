"""
app.py  —  Blog Writing Agent  |  Streamlit Frontend
======================================================
Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

import bwa_backend_updated  # our backend

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS — clean, dark-accent look
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* API key section */
    .api-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    /* Status badges */
    .badge-ok   { color: #16a34a; font-weight: 600; }
    .badge-miss { color: #dc2626; font-weight: 600; }
    .badge-opt  { color: #d97706; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Stream graph updates; fall back to plain invoke."""
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current.update(next(iter(payload.values())))
        else:
            current.update(payload)
    return current


# ─────────────────────────────────────────────
# Markdown renderer (handles local images)
# ─────────────────────────────────────────────

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last: m.start()]
        if before:
            parts.append(("md", before))
        parts.append(("img", f"{m.group('alt')}|||{m.group('src')}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            first_line = nxt.splitlines()[0].strip() if nxt.strip() else ""
            mcap = _CAPTION_RE.match(first_line)
            if mcap:
                caption = mcap.group("cap").strip()
                rest = "\n".join(nxt.splitlines()[1:])
                parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            img_path = Path(src.strip().lstrip("./")).resolve()
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# ─────────────────────────────────────────────
# Past blogs helpers
# ─────────────────────────────────────────────

def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✍️ Blog Writing Agent")
    st.caption("Powered by LangGraph + Groq + Gemini")

    # ── API Keys Section ──────────────────────
    st.divider()
    st.markdown("### 🔑 API Keys")
    st.caption("Keys are used only for this session and never stored.")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Required. Get yours at console.groq.com",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        help="Required for web research. Get yours at tavily.com",
    )
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza... (optional)",
        help="Optional. Only needed for AI image generation. Leave blank to skip images.",
    )

    # Visual key status
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<span class='{'badge-ok' if groq_key else 'badge-miss'}'>{'✓' if groq_key else '✗'} Groq</span>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<span class='{'badge-ok' if tavily_key else 'badge-miss'}'>{'✓' if tavily_key else '✗'} Tavily</span>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<span class='{'badge-ok' if gemini_key else 'badge-opt'}'>{'✓' if gemini_key else '○'} Gemini</span>",
        unsafe_allow_html=True,
    )

    # ── Topic Input ───────────────────────────
    st.divider()
    st.markdown("### 📝 Generate New Blog")

    topic = st.text_area(
        "Topic",
        height=120,
        placeholder="e.g. How self-attention works in transformers",
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    # ── Past Blogs ────────────────────────────
    st.divider()
    st.markdown("### 📂 Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found yet.")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog", use_container_width=True):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_spec": [],
                    "final": md_text,
                }

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown("<div class='main-title'>Blog Writing Agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Generate full technical blog posts with research, planning, and AI images.</div>",
    unsafe_allow_html=True,
)

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs([
    "🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"
])

logs: List[str] = []

# ─────────────────────────────────────────────
# RUN — validate keys, configure backend, stream
# ─────────────────────────────────────────────

if run_btn:
    # Validate required keys
    if not groq_key.strip():
        st.error("❌ Groq API key is required. Add it in the sidebar.")
        st.stop()
    if not tavily_key.strip():
        st.error("❌ Tavily API key is required. Add it in the sidebar.")
        st.stop()
    if not topic.strip():
        st.warning("⚠️ Please enter a topic.")
        st.stop()

    # Inject keys into backend
    bwa_backend_updated.configure_keys(
        groq_api_key=groq_key.strip(),
        tavily_api_key=tavily_key.strip(),
        gemini_api_key=gemini_key.strip(),
    )

    if not gemini_key.strip():
        st.info("ℹ️ No Gemini key provided — image generation will be skipped.")

    # Initial state
    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "need_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "section_md": [],
        "md_place_holders": "",
        "image_spec": [],
        "final": "",
    }

    st.session_state["logs"] = []
    status = st.status("Running pipeline…", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(bwa_backend_updated.app, inputs):
        if kind in ("updates", "values"):
            # Show which node is running
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1:
                candidate = next(iter(payload.values()))
                if isinstance(candidate, dict):
                    node_name = next(iter(payload.keys()))

            if node_name and node_name != last_node:
                status.write(f"➡️ Running node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            # Live summary
            plan_obj = current_state.get("plan")
            task_count = None
            if isinstance(plan_obj, dict):
                task_count = len(plan_obj.get("tasks", []))
            elif hasattr(plan_obj, "tasks"):
                task_count = len(plan_obj.tasks)

            summary = {
                "mode": current_state.get("mode"),
                "need_research": current_state.get("need_research"),
                "queries": (current_state.get("queries") or [])[:5],
                "evidence_count": len(current_state.get("evidence") or []),
                "tasks_planned": task_count,
                "sections_written": len(current_state.get("section_md") or []),
                "images_planned": len(current_state.get("image_spec") or []),
            }
            progress_area.json(summary)
            logs.append(f"[{kind}] {json.dumps(payload, default=str)[:800]}")

        elif kind == "final":
            st.session_state["last_out"] = payload
            st.session_state["logs"].extend(logs)
            status.update(label="✅ Blog generated!", state="complete", expanded=False)
            logs.append("[final] Pipeline complete.")

# ─────────────────────────────────────────────
# RENDER RESULTS
# ─────────────────────────────────────────────

out = st.session_state.get("last_out")

if out:

    # ── Plan tab ─────────────────────────────
    with tab_plan:
        st.subheader("Writing Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan available (loaded from file).")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### {plan_dict.get('blog_title', '')}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Audience:** {plan_dict.get('audience', '')}")
            c2.markdown(f"**Tone:** {plan_dict.get('tone', '')}")

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([{
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "type": t.get("section_type"),
                    "target_words": t.get("target_words"),
                    "code": "✓" if t.get("requires_code") else "",
                    "research": "✓" if t.get("requires_research") else "",
                    "citations": "✓" if t.get("requires_citations") else "",
                } for t in tasks]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Full task details (JSON)"):
                    st.json(tasks)

    # ── Evidence tab ─────────────────────────
    with tab_evidence:
        st.subheader("Research Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence (closed_book mode or Tavily key not set).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append({
                    "title": e.get("title"),
                    "published_at": e.get("published_at"),
                    "source": e.get("source"),
                    "url": e.get("url"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Preview tab ──────────────────────────
    with tab_preview:
        st.subheader("Blog Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            # Resolve title for filename
            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_b:
                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Bundle (MD + Images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    # ── Images tab ───────────────────────────
    with tab_images:
        st.subheader("Generated Images")
        specs = out.get("image_spec") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images were generated for this blog.")
        else:
            if specs:
                with st.expander("Image plan"):
                    st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("`images/` folder exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download All Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # ── Logs tab ─────────────────────────────
    with tab_logs:
        st.subheader("Pipeline Logs")
        all_logs = st.session_state.get("logs", [])
        if not all_logs:
            st.info("No logs yet — run a generation first.")
        else:
            st.text_area(
                "Event log",
                value="\n\n".join(all_logs[-80:]),
                height=520,
                label_visibility="collapsed",
            )
            if st.button("🗑️ Clear logs"):
                st.session_state["logs"] = []
                st.rerun()

else:
    # Empty state — guide the user
    with tab_preview:
        st.markdown("""
        ### 👋 Welcome to the Blog Writing Agent

        **How to get started:**

        1. Add your **Groq** and **Tavily** API keys in the sidebar
        2. Optionally add a **Gemini** key for AI-generated images
        3. Type a topic and click **Generate Blog**

        **Where to get API keys:**
        - Groq → [console.groq.com](https://console.groq.com)
        - Tavily → [tavily.com](https://tavily.com)
        - Gemini → [aistudio.google.com](https://aistudio.google.com)
        """)
