# 🧭 CareerPilot | Autonomous Career Radar & Technical Support Agent

> **Built for the Nebius x NVIDIA Hackathon 2026**
> _Autonomous opportunity discovery, codebase-to-lab fit auditing, and context-grounded outreach synthesis powered by NVIDIA Nemotron on Nebius Token Factory & Tavily Search._

[![Nebius Token Factory](https://img.shields.io/badge/Inference-Nebius%20Token%20Factory-00c996?style=for-the-badge&logo=cloud)](https://nebius.com)
[![NVIDIA Nemotron](https://img.shields.io/badge/Model-Nemotron--3--Ultra--550b-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com)
[![Tavily Search](https://img.shields.io/badge/Live%20Search-Tavily%20API-007A6E?style=for-the-badge)](https://tavily.com)
[![Streamlit UI](https://img.shields.io/badge/Frontend-Streamlit%20Dark-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-38BDF8?style=for-the-badge)](LICENSE)

---

## 🛰️ Executive Overview

Most technical cold outreach fails because students rely on rigid, generic templates that ignore a lab's active research directions. Simultaneously, generic job boards obscure unlisted academic research slots, stealth AI infrastructure roles, and specialized engineering labs.

**CareerPilot** is an autonomous career intelligence agent that closes the divide between student codebases and frontier engineering initiatives. Instead of surface-level keyword matching, CareerPilot parses deep architectural primitives from resumes, queries live university preprints and corporate initiatives using the **Tavily Search API**, and executes a multi-step verification and audit pipeline using **NVIDIA Nemotron-3-Ultra-550b** hosted on **Nebius Token Factory**.

---

## ⚡ Key Features

- **🧬 Multi-Modal Profile Ingestion**: Ingests standard PDF resumes or raw code snippets/coursework notes. Automatically detects non-selectable, image-only PDFs and routes users to text input without crashing downstream parsers.
- **✨ Skillset-Driven Lab Discovery**: Autonomously identifies top-tier research labs, open-source infrastructure teams, and tech groups directly aligned with candidate code patterns (e.g., concurrency models, kernel bypass, distributed pipelines).
- **🎯 Custom Target Radar**: Allows candidates to define any specific target lab, company, or team (e.g., _Stanford Systems Lab_, _vLLM Inference Serving_, _ClickHouse Core_).
- **🌐 Live Web Reconnaissance**: Deploys Tavily Search to scrape active preprints, faculty publications, and current technical requisites in real time.
- **📊 Socratic Fit Cross-Audit**: Computes a grounded synergy score (0–100%), itemizes core technical alignments, isolates critical skill gaps, and features an interactive **Plain-English (ELI5)** toggle for jargon-free feedback.
- **✉️ Grounded Outbound Synthesis**: Generates customized outreach emails that reference real lab papers and codebase mechanics, complete with standard paragraph spacing and a one-click clipboard utility.
- **🔬 Principal Advisor Deep Dive**:
  - **Architectural Fit**: Maps codebase design choices directly to lab bottlenecks.
  - **Day-One Roadmap**: Outlines structured contribution milestones for Days 1–30.
  - **Technical Interview Gauntlet**: Formulates lab-specific technical questions paired with high-signal talking points.
  - **14-Day Study Plan**: Delivers a curriculum tailored to eliminate flagged skill gaps.
- **💻 Interactive Hacker CLI Stream**: A retro-futuristic terminal window showing real-time timestamps, inference telemetry, tool calls, and verification status.

---

## 🏗️ Architecture & Pipeline Flow

```text
               ┌────────────────────────────────────────┐
               │    Candidate Input (PDF or Notes)      │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │   NVIDIA Nemotron-3-Ultra-550b (Nebius Factory)      │
        │     • Language extraction & runtime profiling        │
        │     • Flagship project architecture parsing          │
        └──────────────────────────┬───────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [Custom User Target]                 [Autonomous Skillset Discovery]
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │                Tavily Search Agent                   │
        │    • Live ArXiv preprints & faculty pages            │
        │    • Current team requisites & technical focus       │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │         Nemotron Socratic Fit Cross-Audit            │
        │     • Synergy Scoring & Skill-Gap Isolation          │
        │     • Grounded Cold Outreach Generation              │
        │     • 30-Day Contribution Roadmap & Interview Prep   │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │      CareerPilot UI (Streamlit Dark Cyberpunk)       │
        │     • Live Monospace Execution Terminal              │
        │     • 4-Section Intelligence Report & ELI5 Toggle    │
        └──────────────────────────────────────────────────────┘
```

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/svenmanimala/career-pilot.git
cd career-pilot
```

### 2. Configure Environment & Dependencies

Create and activate an isolated Python virtual environment:

```bash
python3 -m venv venv  # On windows: python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set API Credentials

Create a `.env` file in the root directory:

```env
NEBIUS_API_KEY="your_nebius_token_factory_key"
TAVILY_API_KEY="your_tavily_search_api_key"
```

### 4. Run Application

```bash
streamlit run app.py
```
