# 🚀 Agentic Blog Writing Assistant

A production-style multi-agent AI system that automatically researches, plans, writes, and enriches blog posts using LangGraph, Tavily Search, Groq LLMs, Gemini Image Generation, and Streamlit.

## 📌 Features

* Multi-Agent Workflow using LangGraph
* Intelligent Topic Routing
* Automated Web Research with Tavily
* Blog Planning and Section Generation
* Parallel Writer Agents
* Reducer-Based Content Aggregation
* AI-Generated Images with Gemini
* Markdown Export
* Streamlit User Interface
* Structured Outputs with Pydantic

---

## 🏗️ Architecture

User Input
↓
Router Agent
↓
Research Agent (Tavily)
↓
Planner Agent
↓
Parallel Writer Agents
↓
Reducer Agent
↓
Image Generation Agent (Gemini)
↓
Final Blog Output

---

## 🛠️ Tech Stack

### AI Frameworks

* LangGraph
* LangChain

### LLMs

* Groq (Llama 3.3 70B)
* Google Gemini

### Search & Research

* Tavily Search API

### Frontend

* Streamlit

### Backend

* Python

---

## 📂 Project Structure

├── bwa_backend_updated.py
├── app.py
├── images/
├── requirements.txt
├── .env.example
└── README.md

---

## ⚙️ Setup

### Clone Repository

git clone <repository-url>

### Create Virtual Environment

python -m venv myenv

### Activate Environment

Windows:
myenv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

### Configure API Keys

Create a .env file:

GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
GOOGLE_API_KEY=your_key

### Run Application

streamlit run app.py

---

## 🎯 Example Workflow

1. Enter a blog topic.
2. Router decides if research is needed.
3. Tavily gathers relevant information.
4. Planner creates a structured outline.
5. Parallel agents generate blog sections.
6. Reducer merges all content.
7. Gemini generates supporting images.
8. Final blog is exported as Markdown.

---

## 🚀 Future Improvements

* Long-Term Memory
* Agentic RAG
* SEO Optimization Agent
* Citation Verification Agent
* Multi-Language Support
* PostgreSQL Storage

---

## 👨‍💻 Author
Muhammad Tayyab Sattar

MS Mechanical Engineering Student

Passionate about Agentic AI, Generative AI, and Intelligent Automation Systems.


