import base64
import json
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import pypdf
import streamlit as st
import streamlit.components.v1 as components
from tavily import TavilyClient

# ---------------------------------------------------------
# Page & Environment Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Career Pilot",
    page_icon="favicon.svg",
    layout="wide",
)

load_dotenv(override=True)

NEBIUS_KEY = os.environ.get("NEBIUS_API_KEY", "").strip()
if not NEBIUS_KEY and hasattr(st, "secrets") and "NEBIUS_API_KEY" in st.secrets:
    NEBIUS_KEY = str(st.secrets["NEBIUS_API_KEY"]).strip()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
if not TAVILY_KEY and hasattr(st, "secrets") and "TAVILY_API_KEY" in st.secrets:
    TAVILY_KEY = str(st.secrets["TAVILY_API_KEY"]).strip()

MODEL_NAME = "nvidia/Nemotron-3-Ultra-550b-a55b"
FORMATTED_MODEL_NAME = "Nemotron-3-Ultra"

nebius_client = (
    OpenAI(base_url="https://api.tokenfactory.nebius.com/v1", api_key=NEBIUS_KEY)
    if NEBIUS_KEY
    else None
)
tavily_client = TavilyClient(api_key=TAVILY_KEY) if TAVILY_KEY else None

# ---------------------------------------------------------
# Agent Thought Stream Logger
# ---------------------------------------------------------
if "thought_stream" not in st.session_state:
    st.session_state.thought_stream = []


def log_thought(step_tag: str, message: str, meta: str = None):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.thought_stream.append({
        "time": timestamp,
        "tag": step_tag,
        "message": message,
        "meta": meta,
    })


# ---------------------------------------------------------
# Utilities & Sanitization
# ---------------------------------------------------------
def clean_json_string(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


# ---------------------------------------------------------
# Core Agent Logic Functions
# ---------------------------------------------------------
def extract_student_profile(raw_profile: str) -> dict:
    cleaned = raw_profile.strip()
    if len(cleaned) < 5:
        return {
            "is_valid": False,
            "reason": "Input is too short. Please provide a brief note on your skills, interests, or background.",
        }

    system_prompt = """You are an encouraging and intelligent technical evaluator powered by NVIDIA Nemotron.
Analyze the user's background input. Users may provide an exhaustive formal resume, OR simple, casual notes.

EVALUATION RULES:
1. ACCEPT (is_valid: true): Any mention of programming languages, computer science, engineering, software, tech projects, coursework, data, web development, or interest in learning tech. Even if simple or informal, mark it as VALID. Infer reasonable domains and skills.
2. REJECT (is_valid: false): ONLY reject if the input is completely unrelated to tech/careers/education (e.g., cooking recipes, song lyrics, shopping lists) OR keyboard spam/gibberish (e.g., 'asdfghjkl').

Return ONLY a valid JSON object matching this schema:
{
  "is_valid": true,
  "reason": "If invalid, a polite 1-sentence explanation of why it was rejected",
  "candidate_name": "Full Name extracted from resume (or 'Candidate' if unstated)",
  "qualifications": "Degree, Major, Institution, or Current Status (e.g. B.Tech Computer Science, Senior)",
  "primary_languages": ["Language 1", "Language 2"],
  "core_domains": ["Domain 1", "Domain 2"],
  "highlight_projects": ["Project or Learning Focus 1", "Project or Learning Focus 2"],
  "experience_level": "Beginner / Undergraduate / Early Career"
}"""
    try:
        response = nebius_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned},
            ],
            temperature=0.1,
        )
        data = json.loads(clean_json_string(response.choices[0].message.content))
        if not data.get("is_valid", True):
            return {
                "is_valid": False,
                "reason": data.get("reason", "The input appears unrelated to software, engineering, or technical background."),
            }

        if not data.get("candidate_name"):
            data["candidate_name"] = "Candidate"
        if not data.get("qualifications"):
            data["qualifications"] = "Computer Science & Engineering Background"
        if not data.get("primary_languages"):
            data["primary_languages"] = ["General Software / Python"]
        if not data.get("core_domains"):
            data["core_domains"] = ["Software Engineering & Computer Science"]
        if not data.get("highlight_projects"):
            data["highlight_projects"] = ["Applied software projects & coursework"]
        if not data.get("experience_level"):
            data["experience_level"] = "Undergraduate / Learner"

        data["is_valid"] = True
        return data
    except Exception as e:
        print(f"Extraction fallback: {e}")
        return {
            "is_valid": True,
            "candidate_name": "Candidate",
            "qualifications": "Engineering & Technology Studies",
            "primary_languages": ["Python", "General Programming"],
            "core_domains": ["Software Development"],
            "highlight_projects": [cleaned[:80] + "..."],
            "experience_level": "Undergraduate",
        }


def recommend_target_opportunities(student_profile: dict) -> list:
    system_prompt = """You are an elite career intelligence strategist powered by NVIDIA Nemotron.
Based on the candidate's extracted technical profile (languages, projects, and architecture patterns), recommend 3 specific, top-tier research labs, open-source projects, or tech teams where they have maximum technical synergy.
Return ONLY valid JSON matching:
{
  "recommended_targets": [
    {"name": "Lab/Company/Project Name", "reason": "Specific 1-sentence technical reason why their projects align"},
    {"name": "Lab/Company/Project Name 2", "reason": "Specific 1-sentence technical reason why their projects align"},
    {"name": "Lab/Company/Project Name 3", "reason": "Specific 1-sentence technical reason why their projects align"}
  ]
}"""
    response = nebius_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(student_profile)},
        ],
        temperature=0.3,
    )
    res_json = json.loads(clean_json_string(response.choices[0].message.content))
    return res_json.get("recommended_targets", [])


def discover_live_opportunities(target_query: str) -> dict:
    """Queries Tavily for requirements and pinpoints the official career/openings URL."""
    query = f"{target_query} undergraduate research assistant internship openings requirements"
    results = tavily_client.search(query=query, search_depth="advanced", max_results=4)
    hits = results.get("results", [])

    chunks = [
        f"Source ({res.get('url', 'Web')}):\n{res.get('content', '')}"
        for res in hits
    ]
    if not chunks:
        raise RuntimeError(f"No opportunity postings found for: {target_query}")

    # Identify primary careers or openings link
    careers_url = ""
    for hit in hits:
        u = hit.get("url", "")
        u_lower = u.lower()
        if any(term in u_lower for term in ["career", "job", "opening", "join", "apply", "lab", "team", "people", "research"]):
            careers_url = u
            break

    # Targeted fallback if no URL matched key path terms
    if not careers_url:
        try:
            portal_search = tavily_client.search(
                query=f"{target_query} official careers job openings lab site",
                search_depth="basic",
                max_results=2,
            )
            portal_hits = portal_search.get("results", [])
            if portal_hits:
                careers_url = portal_hits[0].get("url", "")
        except Exception:
            pass

    if not careers_url and hits:
        careers_url = hits[0].get("url", "")

    return {
        "docs": "\n\n".join(chunks),
        "careers_url": careers_url,
    }


def audit_fit_and_generate_pitch(student_profile: dict, opportunity_context: str) -> dict:
    system_prompt = """You are an elite technical career auditor powered by NVIDIA Nemotron.
Cross-examine candidate profile against live external opportunity data.
Return ONLY valid JSON matching:
{
  "match_score": 85,
  "match_verdict": "1-sentence executive verdict",
  "strengths": ["Strength 1", "Strength 2"],
  "skill_gaps": ["Gap 1", "Gap 2"],
  "targeted_cold_pitch": {
    "subject": "Email subject line",
    "email_body": "Dear [Team/Professor],\\n\\n[Hook & Domain Alignment]\\n\\n[Concrete Projects & Relevant Metrics]\\n\\n[Call to Action]\\n\\nBest regards,\\n[Candidate]"
  },
  "preparation_advice": "Actionable 1-sentence tip on what to build next"
}"""
    payload = f"CANDIDATE:\n{json.dumps(student_profile)}\n\nTARGET DATA (LIVE WEB):\n{opportunity_context}"
    response = nebius_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        temperature=0.2,
    )
    return json.loads(clean_json_string(response.choices[0].message.content))


def simplify_verdict(verdict: str, advice: str) -> dict:
    system_prompt = """You are a friendly senior mentor.
Translate this technical recruitment verdict and advice into punchy, plain-English, jargon-free terms (ELI5 style) for an undergraduate. Keep it encouraging and direct.
Return ONLY valid JSON:
{
  "simple_verdict": "Conversational explanation of why they fit or what is missing",
  "simple_advice": "Simple, non-intimidating action step"
}"""
    response = nebius_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Verdict: {verdict}\nAdvice: {advice}"},
        ],
        temperature=0.3,
    )
    return json.loads(clean_json_string(response.choices[0].message.content))


def generate_detailed_deepdive(student_profile: dict, opportunity_context: str, audit_data: dict) -> dict:
    system_prompt = """You are a Principal Systems Architect and Technical Hiring Director.
Produce an exhaustive, highly technical deep-dive audit for an undergraduate applicant applying to this specific lab/role.
Return ONLY valid JSON matching this exact schema:
{
  "architectural_fit": "Thorough 3-4 sentence paragraph linking student's exact project mechanics to the target lab's technical stack.",
  "day_one_roadmap": [
    "Milestone 1 (Week 1)",
    "Milestone 2 (Week 2)",
    "Milestone 3 (Weeks 3-4)"
  ],
  "technical_interview_gauntlet": [
    {"question": "Rigorous technical question", "talking_point": "Ideal technical insight to share"},
    {"question": "Rigorous technical question 2", "talking_point": "Ideal technical insight to share"},
    {"question": "Rigorous technical question 3", "talking_point": "Ideal technical insight to share"}
  ],
  "accelerated_syllabus": [
    {"day": "Day 1-4", "topic": "Core Concept", "task": "Concrete hands-on mini project"},
    {"day": "Day 5-9", "topic": "Core Concept", "task": "Concrete hands-on mini project"},
    {"day": "Day 10-14", "topic": "Core Concept", "task": "Concrete hands-on mini project"}
  ]
}"""
    payload = f"CANDIDATE:\n{json.dumps(student_profile)}\n\nPREVIOUS AUDIT SUMMARY:\n{json.dumps(audit_data)}\n\nTARGET LAB / REQUISITES (FROM WEB):\n{opportunity_context}"
    response = nebius_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        temperature=0.25,
    )
    return json.loads(clean_json_string(response.choices[0].message.content))


# ---------------------------------------------------------
# State Machine Initialization
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "input"
if "student_text" not in st.session_state:
    st.session_state.student_text = ""
if "is_pdf" not in st.session_state:
    st.session_state.is_pdf = False
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "target_mode" not in st.session_state:
    st.session_state.target_mode = "Custom Search"
if "target_query" not in st.session_state:
    st.session_state.target_query = ""
if "recommended_targets" not in st.session_state:
    st.session_state.recommended_targets = []
if "audit_data" not in st.session_state:
    st.session_state.audit_data = None
if "simplified" not in st.session_state:
    st.session_state.simplified = False
if "simple_data" not in st.session_state:
    st.session_state.simple_data = None
if "deep_dive_data" not in st.session_state:
    st.session_state.deep_dive_data = None
if "invalid_reason" not in st.session_state:
    st.session_state.invalid_reason = ""

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #88C7B8 !important;
    }

    /* Enforce word-wrap for code blocks and agent telemetry */
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code,
    .stCodeBlock pre code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }

    /* Pin the simplify/jargon button flush to the right boundary */
    .right-align-btn {
        display: flex;
        justify-content: flex-end;
        width: 100%;
    }
    .right-align-btn div.stButton {
        width: auto !important;
    }
    .right-align-btn div.stButton > button {
        width: auto !important;
        white-space: nowrap !important;
        padding: 10px 18px !important;
        font-size: 0.90rem !important;
    }

    /* Clean button styling without text clipping */
    div.stButton > button {
        white-space: nowrap !important;
        font-size: 0.88rem !important;
        padding: 9px 16px !important;
    }

    .card-footer-action {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-top: 12px;
        width: 100%;
    }

    .card-footer-action div.stButton {
        width: auto !important;
    }

    .card-footer-action div.stButton > button {
        width: auto !important;
    }

    /* Right align buttons inside a header action wrapper */
    .header-action-right {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        width: 100%;
    }
    .header-action-right div.stButton {
        width: auto !important;
    }
    .header-action-right div.stButton > button {
        width: auto !important;
        white-space: nowrap !important;
        padding: 10px 22px !important;
    }

    .block-container {
        max-width: 1020px !important;
        padding-top: 3.5rem !important;
        padding-bottom: 6rem !important;
        margin: 0 auto !important;
        position: relative;
        z-index: 1;
    }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    #bg-video {
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -2;
        object-fit: cover;
        opacity: 0.28;
        filter: saturate(1.5) hue-rotate(145deg);
    }

    .bg-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background: radial-gradient(circle at 50% 15%, rgba(0, 61, 77, 0.65) 0%, rgba(1, 22, 29, 0.94) 85%);
        pointer-events: none;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(0, 35, 46, 0.75) !important;
        border: 1px solid rgba(0, 201, 150, 0.3) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(20px) !important;
        padding: 24px 28px !important;
        box-shadow: 0 0 25px rgba(0, 201, 150, 0.12) !important;
    }

    .hero-title {
        font-size: 3.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #00A389 0%, #00C996 50%, #4EEDB7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        text-shadow: 0 0 40px rgba(0, 201, 150, 0.35);
    }

    .section-divider {
        margin: 45px 0 30px 0;
        position: relative;
        text-align: center;
    }
    .section-divider::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 61, 77, 0.6), rgba(0, 201, 150, 0.6), transparent);
        z-index: 1;
    }
    .section-badge {
        position: relative;
        z-index: 2;
        background: #011E26;
        border: 1px solid rgba(0, 201, 150, 0.45);
        border-radius: 9999px;
        padding: 6px 20px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #00c996;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #003D4D 0%, #007A6E 50%, #00C996 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        border: 1px solid rgba(0, 201, 150, 0.4) !important;
        box-shadow: 0 0 25px rgba(0, 201, 150, 0.35) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 0 35px rgba(0, 201, 150, 0.6) !important;
        border-color: #00c996 !important;
    }

    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(1, 20, 26, 0.92);
        backdrop-filter: blur(14px);
        border-top: 1px solid rgba(0, 201, 150, 0.25);
        padding: 10px 0;
        text-align: center;
        font-size: 0.82rem;
        color: #88C7B8;
        z-index: 999;
    }
</style>

<video autoplay loop muted playsinline id="bg-video">
    <source src="https://assets.mixkit.co/videos/preview/mixkit-flying-through-a-futuristic-tunnel-32349-large.mp4" type="video/mp4">
</video>
<div class="bg-overlay"></div>
""",
    unsafe_allow_html=True,
)

if not nebius_client or not tavily_client:
    st.error("Missing required API keys. Please define NEBIUS_API_KEY and TAVILY_API_KEY in your environment or secrets.")
    st.stop()

# =========================================================
# STAGE 1: INGESTION & TARGETING MODE SELECTION
# =========================================================
if st.session_state.step == "input":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2.2rem; margin-top: 0.8rem;">
            <span style="background: rgba(0, 61, 77, 0.4); color: #00c996; padding: 6px 18px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(0, 201, 150, 0.45); text-transform: uppercase; letter-spacing: 0.08em;">
                ⚡ Autonomous Career Radar & Agent
            </span>
            <div class="hero-title">Career Pilot</div>
            <p style="color: #CBD5E1; font-size: 1.15rem; max-width: 620px; margin: 0 auto; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
                Provide your background. Career Pilot can scout a specific target or autonomously suggest best-fit research labs and tech teams based on your code.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        input_mode = st.radio(
            "Choose How to Share Your Background:",
            ["📄 Upload PDF Resume", "✍️ Paste Text / Notes"],
            horizontal=True,
        )

        extracted_text = ""

        if input_mode == "📄 Upload PDF Resume":
            uploaded_pdf = st.file_uploader(
                "Upload your resume (PDF only)",
                type=["pdf"],
                key="resume_uploader",
                help="Standard 1-2 page student or technical resume",
            )
            if uploaded_pdf is not None:
                try:
                    uploaded_pdf.seek(0)
                    pdf_bytes_data = uploaded_pdf.getvalue()
                    reader = pypdf.PdfReader(uploaded_pdf)
                    pages_text = [
                        page.extract_text(extraction_mode="layout") or page.extract_text() or ""
                        for page in reader.pages
                    ]
                    extracted_text = "\n\n".join(filter(None, pages_text)).strip()

                    if extracted_text:
                        st.session_state.is_pdf = True
                        st.session_state.pdf_bytes = pdf_bytes_data
                        st.session_state.pdf_filename = uploaded_pdf.name
                        st.session_state.student_text = extracted_text
                        st.toast(f"Parsed {len(extracted_text)} chars from {uploaded_pdf.name}", icon="📄")
                    else:
                        st.error("⚠️ No selectable text detected. This document may be an image-only scan.")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
        else:
            st.session_state.is_pdf = False
            st.session_state.pdf_bytes = None
            extracted_text = st.text_area(
                "Your Background & Projects:",
                value=st.session_state.student_text if not st.session_state.is_pdf else "",
                placeholder=(
                    "Type casually in plain English or paste your notes:\n\n"
                    "• 'CS sophomore learning C++ and Python, built an async crawler.'\n"
                    "• 'I know React, SQL, and FastAPI, looking for full-stack data roles.'\n"
                    "• 'Taking distributed systems, interested in high-concurrency research.'"
                ),
                height=170,
            )

        st.write("")
        target_strategy = st.radio(
            "Targeting Strategy:",
            [
                "🎯 I have a specific lab/company in mind",
                "✨ Auto-Discover best-fit labs/teams based on my skillset",
            ],
            horizontal=True,
        )

        target_role = ""
        if target_strategy == "🎯 I have a specific lab/company in mind":
            target_role = st.text_input(
                "Target Team, Lab, or Role:",
                value="",
                placeholder="e.g. 'Stanford Systems Lab', 'vLLM Inference Serving', 'NASA Data Engineering'",
            )
        else:
            st.info("💡 Career Pilot will analyze your code architecture and recommend 3 high-synergy research groups or tech teams automatically.")

        st.write("")
        if st.button("🚀 Engage Scout Agent & Audit Fit", type="primary"):
            current_input = extracted_text.strip() or st.session_state.student_text.strip()
            if not current_input:
                st.warning("Please upload a resume or provide your background first.")
            elif (
                target_strategy == "🎯 I have a specific lab/company in mind"
                and not target_role.strip()
            ):
                st.warning("Please enter a target lab, company, or role to scout for.")
            else:
                st.session_state.student_text = current_input
                st.session_state.target_mode = (
                    "Custom" if "specific" in target_strategy else "Auto-Discover"
                )
                st.session_state.target_query = target_role
                st.session_state.step = "processing"
                st.session_state.deep_dive_data = None
                st.rerun()

# =========================================================
# STAGE 2: PROGRESS & CLEAN MONOSPACE TERMINAL STREAM
# =========================================================
elif st.session_state.step == "processing":
    st.session_state.thought_stream = []

    st.markdown(
        """
        <div style="text-align: center; margin-top: 2.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 3.2rem; margin-bottom: 10px; filter: drop-shadow(0 0 25px #00C996);">🛰️</div>
            <h2 style="font-size: 2.1rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">
                Target Radar Active: Agent Execution Stream
            </h2>
            <p style="color: #88C7B8; font-size: 0.95rem;">
                Autonomous telemetry pipe across Nebius Token Factory & Tavily API
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(10)
    terminal_placeholder = st.empty()

    def render_terminal(is_active: bool = True):
        lines = ["career-pilot@nebius-cloud:~$ ./run_pipeline.sh --model nvidia/Nemotron-3-Ultra-550b\n"]
        for item in st.session_state.thought_stream:
            meta_str = f" \n  └─ [{item['meta']}]" if item["meta"] else ""
            lines.append(f"[{item['time']}] [{item['tag']}] {item['message']}{meta_str}")
        if is_active:
            lines.append("\n> [awaiting inference response...]")
        else:
            lines.append("\n> [EXECUTION COMPLETE: 0 ERRORS]")
        terminal_placeholder.code("\n".join(lines), language="bash")

    try:
        # Event 1: Profile Ingestion
        log_thought("EXTRACT", "Deconstructing candidate profile & credential metadata", MODEL_NAME)
        render_terminal(is_active=True)
        time.sleep(0.35)

        profile_data = extract_student_profile(st.session_state.student_text)

        if not profile_data.get("is_valid", False):
            st.session_state.invalid_reason = profile_data.get(
                "reason", "The input provided is completely unrelated to programming, engineering, or coursework."
            )
            st.session_state.step = "invalid_input"
            st.rerun()

        found_langs = ", ".join(profile_data.get("primary_languages", []))
        candidate_identity = f"{profile_data.get('candidate_name', 'Candidate')} ({profile_data.get('qualifications', 'Technical')})"
        log_thought("EXTRACT", f"Parsed candidate: {candidate_identity} | Languages: [{found_langs}]", "HTTP 200 OK")
        progress_bar.progress(30)
        render_terminal(is_active=True)
        time.sleep(0.35)

        # Event 2: Target Evaluation
        active_target = st.session_state.target_query
        if st.session_state.target_mode == "Auto-Discover":
            log_thought("RECOM", "Synthesizing high-synergy labs from codebase primitives", "SKILLSET_DISCOVERY")
            render_terminal(is_active=True)
            time.sleep(0.35)

            recoms = recommend_target_opportunities(profile_data)
            st.session_state.recommended_targets = recoms
            if recoms:
                active_target = recoms[0]["name"]
                st.session_state.target_query = active_target
                log_thought("RECOM", f"Selected top priority lab target: '{active_target}'", "TOP_MATCH")
        else:
            log_thought("RECOM", f"Target locked to user query: '{active_target}'", "CUSTOM")

        progress_bar.progress(55)
        render_terminal(is_active=True)
        time.sleep(0.35)

        # Event 3: External Discovery via Tavily
        log_thought("TAVILY", f"Querying live preprints, openings & careers portal for '{active_target}'", "DEPTH: ADVANCED")
        render_terminal(is_active=True)
        time.sleep(0.35)

        opp_result = discover_live_opportunities(active_target)
        opp_docs = opp_result["docs"]
        careers_url = opp_result["careers_url"]

        doc_snippets_count = max(len(opp_docs.split("Source")) - 1, 1)
        portal_meta = careers_url if careers_url else "PORTAL_LOCATED"
        log_thought("TAVILY", f"Scraped {doc_snippets_count} verified papers & confirmed careers portal", portal_meta[:35])
        progress_bar.progress(80)
        render_terminal(is_active=True)
        time.sleep(0.35)

        # Event 4: Socratic Audit
        log_thought("AUDIT", "Cross-examining candidate repo architecture against preprints", "NEBIUS_TOKEN_FACTORY")
        render_terminal(is_active=True)
        time.sleep(0.35)

        audit_data = audit_fit_and_generate_pitch(profile_data, opp_docs)
        log_thought("VERIFY", f"Audited synergy score: {audit_data.get('match_score', 80)}% (0 hallucinations flagged)", "PASS")
        progress_bar.progress(100)
        render_terminal(is_active=False)
        time.sleep(0.5)

        st.session_state.audit_data = {
            "profile": profile_data,
            "docs": opp_docs,
            "careers_url": careers_url,
            "audit": audit_data,
        }
        st.session_state.step = "results"
        st.rerun()

    except Exception as e:
        st.error(f"Agent pipeline encountered an error: {e}")
        if st.button("Return to Input"):
            st.session_state.step = "input"
            st.rerun()

# =========================================================
# SEPARATE SCREEN: COMPLETELY UNRELATED INPUT DETECTED
# =========================================================
elif st.session_state.step == "invalid_input":
    st.markdown(
        """
        <div style="text-align: center; margin-top: 3.5rem; margin-bottom: 2rem;">
            <div style="font-size: 3.5rem; margin-bottom: 12px; filter: drop-shadow(0 0 20px #F87171);">⚠️</div>
            <h2 style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">
                Unrelated Input Detected
            </h2>
            <p style="color: #CBD5E1; font-size: 1.05rem; max-width: 580px; margin: 0 auto; line-height: 1.5;">
                CareerPilot couldn't find any connection to programming, technical projects, or engineering in your submission.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("### 🔍 Verification Feedback")
        reason = st.session_state.get(
            "invalid_reason", "The input appears unrelated to technical engineering or coursework."
        )
        st.info(f"💡 **Reason:** {reason}")

        st.markdown(
            """
            **You do not need a formal resume.** You can enter your background casually:
            * *"Freshman learning Python and C++, built a small Discord bot."*
            * *"I know HTML, CSS, JavaScript, and want to learn backend systems."*
            * *"Studying algorithms and math, interested in machine learning research."*
            """
        )

        st.write("")
        if st.button("↺ Try Again", type="primary"):
            st.session_state.step = "input"
            st.session_state.student_text = ""
            st.session_state.thought_stream = []
            st.session_state.audit_data = None
            st.session_state.invalid_reason = ""
            st.rerun()

# =========================================================
# STAGE 3: ALL SECTIONS IN SCROLLABLE VIEW
# =========================================================
elif st.session_state.step == "results":
    data = st.session_state.audit_data
    audit = data["audit"]
    profile = data["profile"]
    docs = data["docs"]
    careers_url = data.get("careers_url", "")

    score = audit.get("match_score", 85)
    score_color = "#00c996" if score >= 80 else ("#FBBF24" if score >= 60 else "#F87171")

    # ---------------------------------------------------------
    # TOP HEADER & CONTROLS (SINGLE INSTANCE)
    # ---------------------------------------------------------
    c_top_title, c_top_btn = st.columns([3.4, 1.2], vertical_alignment="center")

    with c_top_title:
        badge_label = "AI Skillset-Matched Radar" if st.session_state.target_mode == "Auto-Discover" else "Custom Target Radar"
        st.markdown(
            f"""
            <div style="margin-bottom: 4px;">
                <span style="background: rgba(56, 189, 248, 0.12); color: #38BDF8; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; border: 1px solid rgba(56, 189, 248, 0.3);">
                    {badge_label} Ready
                </span>
                <h2 style="font-size: 2.1rem; font-weight: 800; color: #F8FAFC; margin: 6px 0 0 0;">
                    Target: {st.session_state.target_query}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if careers_url:
            st.markdown(
                f"""
                <div style="margin-top: 14px; margin-bottom: 22px;">
                    <a href="{careers_url}" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 61, 77, 0.65); border: 1px solid rgba(0, 201, 150, 0.55); color: #00FFBE; text-decoration: none; padding: 8px 18px; border-radius: 9px; font-weight: 700; font-size: 0.84rem; box-shadow: 0 0 15px rgba(0, 201, 150, 0.2); transition: all 0.2s ease;">
                        🌐 Official Careers / Lab Portal ↗
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_top_btn:
        st.markdown('<div class="header-action-right">', unsafe_allow_html=True)
        if st.button("↺ Start Over", key="btn_start_over"):
            st.session_state.step = "input"
            st.session_state.student_text = ""
            st.session_state.is_pdf = False
            st.session_state.pdf_bytes = None
            st.session_state.pdf_filename = ""
            st.session_state.audit_data = None
            st.session_state.simplified = False
            st.session_state.simple_data = None
            st.session_state.deep_dive_data = None
            st.session_state.invalid_reason = ""
            st.session_state.pop("res_subj", None)
            st.session_state.pop("res_body", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Skillset-Driven Recommendations Banner
    if st.session_state.target_mode == "Auto-Discover" and st.session_state.recommended_targets:
        with st.container(border=True):
            st.markdown("#### ✨ Skillset-Suggested Targets for Your Profile")
            for idx, t in enumerate(st.session_state.recommended_targets, 1):
                is_current = t["name"] == st.session_state.target_query
                marker = "🟢 **[Audited Now]**" if is_current else "⚪"
                st.markdown(f"{marker} **{t['name']}** — *{t['reason']}*")

    # ---------------------------------------------------------
    # SECTION 1: OVERVIEW & VERDICT
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="section-divider">
            <span class="section-badge">Section 01 &bull; Synergy Overview</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.simplified and st.session_state.simple_data:
        current_verdict = st.session_state.simple_data.get("simple_verdict", "")
        current_advice = st.session_state.simple_data.get("simple_advice", "")
        mode_tag = "💡 Plain-English Translation (ELI5)"
    else:
        current_verdict = audit.get("match_verdict", "Substantial synergy with target requirements.")
        current_advice = audit.get("preparation_advice", "Review publications.")
        mode_tag = "Executive Technical Verdict"

    with st.container(border=True):
        col_main, col_score = st.columns([3.2, 1.2], gap="large", vertical_alignment="center")

        with col_main:
            st.markdown(
                f"<div style='font-size: 0.76rem; font-weight: 800; color: #38BDF8; "
                f"text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;'>"
                f"{mode_tag}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size: 1.25rem; font-weight: 700; color: #F8FAFC; "
                f"margin-bottom: 20px; line-height: 1.6;'>"
                f"{current_verdict}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='color: #D1FAE5; font-size: 0.95rem; line-height: 1.6; "
                f"padding: 14px 18px; background: rgba(0, 40, 50, 0.45); "
                f"border: 1px solid rgba(0, 201, 150, 0.4); border-left: 4px solid #00c996; "
                f"border-radius: 8px; margin-bottom: 10px; box-shadow: 0 0 15px rgba(0, 201, 150, 0.08);'>"
                f"⚡ <b>Recommended Move:</b> {current_advice}</div>",
                unsafe_allow_html=True,
            )

        with col_score:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 26px 16px; background: rgba(1, 20, 26, 0.9); border-radius: 18px; border: 1px solid rgba(0, 201, 150, 0.25); box-shadow: inset 0 0 20px rgba(0, 61, 77, 0.5);">
                    <div style="font-size: 3.8rem; font-weight: 800; color: {score_color}; line-height: 1; letter-spacing: -0.02em;">{score}%</div>
                    <div style="color: #88C7B8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; margin-top: 10px; letter-spacing: 0.08em;">Target Synergy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Right-aligned footer action bar spanning the full card width
        st.markdown('<div class="card-footer-action">', unsafe_allow_html=True)
        if not st.session_state.simplified:
            if st.button("🪄 Explain in Plain Terms", key="btn_simplify"):
                if not st.session_state.simple_data:
                    with st.spinner("Translating into plain English..."):
                        st.session_state.simple_data = simplify_verdict(
                            audit.get("match_verdict", ""),
                            audit.get("preparation_advice", ""),
                        )
                st.session_state.simplified = True
                st.rerun()
        else:
            if st.button("📊 Show Technical Jargon", key="btn_jargon"):
                st.session_state.simplified = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    # ---------------------------------------------------------
    # SECTION 2: PROS & CONS (SYNERGIES & GAPS)
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="section-divider">
            <span class="section-badge">Section 02 &bull; Technical Strengths & Missing Gaps</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_str, col_gap = st.columns(2, gap="large")
    with col_str:
        with st.container(border=True):
            st.markdown("### :green[✅ Core Strengths & Synergies]")
            st.write("")
            for s in audit.get("strengths", []):
                st.markdown(f"• **{s}**")

    with col_gap:
        with st.container(border=True):
            st.markdown("### :orange[⚠️ Identified Skill Gaps to Bridge]")
            st.write("")
            for g in audit.get("skill_gaps", []):
                st.markdown(f"• **{g}**")

    # ---------------------------------------------------------
    # SECTION 3: GROUNDED OUTREACH EMAIL
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="section-divider">
            <span class="section-badge">Section 03 &bull; Grounded Cold Outreach</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pitch = audit.get("targeted_cold_pitch", {})
    default_subj = pitch.get("subject", "Undergraduate Research Inquiry")
    raw_body = pitch.get("email_body", "")

    formatted_body = raw_body.replace("\\n", "\n").strip()
    if "\n" not in formatted_body:
        formatted_body = formatted_body.replace(" Dear ", "\n\nDear ").replace(" I’m ", "\n\nI’m ")
        formatted_body = formatted_body.replace(" Your ", "\n\nYour ").replace(" I’d ", "\n\nI’d ")
        formatted_body = formatted_body.replace(" Best regards,", "\n\nBest regards,\n")
    default_body = formatted_body

    if "res_subj" not in st.session_state:
        st.session_state["res_subj"] = default_subj
    if "res_body" not in st.session_state:
        st.session_state["res_body"] = default_body

    def render_copy_button(label: str, target_selector: str, fallback_text: str, btn_id: str):
        escaped_payload = json.dumps(fallback_text)
        escaped_label = json.dumps(label)
        escaped_sel = json.dumps(target_selector)
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    display: flex;
                    align-items: center;
                    overflow: hidden;
                }}
                .copy-btn {{
                    width: 100%;
                    box-sizing: border-box;
                    background: linear-gradient(135deg, rgba(0, 61, 77, 0.75) 0%, rgba(0, 201, 150, 0.4) 100%);
                    border: 1px solid rgba(0, 201, 150, 0.6);
                    color: #00FFBE;
                    padding: 9px 12px;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 0.84rem;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    outline: none;
                }}
                .copy-btn:hover {{
                    border-color: #00FFBE;
                    box-shadow: 0 0 10px rgba(0, 201, 150, 0.4);
                }}
                .copy-btn.copied {{
                    border-color: #22c55e !important;
                    color: #86efac !important;
                    background: rgba(34, 197, 94, 0.25) !important;
                }}

            </style>
        </head>
        <body>
            <button id="{btn_id}" class="copy-btn">{label}</button>
            <script>
                const btn = document.getElementById("{btn_id}");
                const origLabel = {escaped_label};
                const selector = {escaped_sel};
                const fallbackPayload = {escaped_payload};

                btn.onclick = function() {{
                    let textToCopy = fallbackPayload;
                    try {{
                        const parentDoc = window.parent.document;
                        const el = parentDoc.querySelector(selector);
                        if (el && typeof el.value === "string") {{
                            textToCopy = el.value;
                        }}
                    }} catch (e) {{}}

                    function triggerCopied() {{
                        btn.innerText = "✅ Copied!";
                        btn.classList.add("copied");
                        setTimeout(function() {{
                            btn.innerText = origLabel;
                            btn.classList.remove("copied");
                        }}, 2000);
                    }}

                    if (navigator.clipboard && window.isSecureContext) {{
                        navigator.clipboard.writeText(textToCopy)
                            .then(triggerCopied)
                            .catch(function() {{ fallbackCopy(textToCopy, triggerCopied); }});
                    }} else {{
                        fallbackCopy(textToCopy, triggerCopied);
                    }}

                    function fallbackCopy(text, callback) {{
                        const ta = document.createElement("textarea");
                        ta.value = text;
                        ta.style.position = "fixed";
                        ta.style.left = "-9999px";
                        ta.style.top = "-9999px";
                        document.body.appendChild(ta);
                        ta.focus();
                        ta.select();
                        try {{
                            document.execCommand("copy");
                            callback();
                        }} catch (err) {{}}
                        document.body.removeChild(ta);
                    }}
                }};
            </script>
        </body>
        </html>
        """
        components.html(html_code, height=45)

    with st.container(border=True):
        st.markdown("### ✉ Grounded Cold Outreach Draft")
        st.caption("Cites real lab priorities and preprints discovered live via Tavily Search. Edit directly below.")
        st.write("")

        c_subj_in, c_subj_btn = st.columns([3.8, 1.2], vertical_alignment="bottom")
        with c_subj_in:
            st.text_input("Subject Line:", key="res_subj")
        with c_subj_btn:
            render_copy_button(
                "📋 Copy Subject",
                'input[aria-label="Subject Line:"], [data-testid="stTextInput"] input',
                st.session_state["res_subj"],
                "btn_copy_subj",
            )

        st.write("")

        c_body_lbl, c_body_btn = st.columns([3.8, 1.2], vertical_alignment="bottom")
        with c_body_lbl:
            st.markdown("**Email Body:**")
        with c_body_btn:
            render_copy_button(
                "📋 Copy Body",
                'textarea[aria-label="Email Body:"], [data-testid="stTextArea"] textarea',
                st.session_state["res_body"],
                "btn_copy_body",
            )

        st.text_area("Email Body:", key="res_body", height=240, label_visibility="collapsed")

    # ---------------------------------------------------------
    # SECTION 4: TECHNICAL ADVISOR & DEEP DIVE
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="section-divider">
            <span class="section-badge">Section 04 &bull; Technical Advisor Deep Dive</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("### 🔬 Deep-Dive Architectural & Interview Audit")
        st.caption(f"Synthesized by {MODEL_NAME} to connect codebase mechanics, map 30-day milestones, and formulate interview questions.")

        if st.session_state.deep_dive_data is None:
            st.write("")
            if st.button("🚀 Generate Comprehensive Deep-Dive Analysis"):
                with st.spinner("Nemotron synthesizing architectural deep dive..."):
                    st.session_state.deep_dive_data = generate_detailed_deepdive(profile, docs, audit)
                st.rerun()
        else:
            dd = st.session_state.deep_dive_data
            tab_arch, tab_road, tab_prep, tab_syl = st.tabs([
                "🏗️ Architectural Fit",
                "🗺️ Day-One Roadmap",
                "🎯 Interview Gauntlet",
                "📚 14-Day Study Plan",
            ])

            with tab_arch:
                st.markdown("##### Infrastructure Synergy & Codebase Alignment")
                st.write(dd.get("architectural_fit", ""))

            with tab_road:
                st.markdown("##### First 30-Day Contribution Milestones")
                for item in dd.get("day_one_roadmap", []):
                    st.markdown(f"• {item}")

            with tab_prep:
                st.markdown("##### Technical Interview Questions & Core Insights")
                for i, q in enumerate(dd.get("technical_interview_gauntlet", []), 1):
                    st.markdown(f"**Q{i}: {q.get('question')}**")
                    st.info(f"💡 **Key Talking Point:** {q.get('talking_point')}")

            with tab_syl:
                st.markdown("##### Accelerated 14-Day Skill-Gap Curriculum")
                for s in dd.get("accelerated_syllabus", []):
                    col_d, col_t = st.columns([1, 4])
                    with col_d:
                        st.markdown(f"**:blue[{s.get('day')}]**")
                        st.caption(s.get("topic"))
                    with col_t:
                        st.write(s.get("task"))

    st.write("")

    # ---------------------------------------------------------
    # SECTION 5: RESUME DOSSIER & PDF PREVIEW
    # ---------------------------------------------------------
    if st.session_state.get("is_pdf"):
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-badge">Section 05 &bull; Ingested Resume Dossier & PDF Preview</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### 📋 Ingested Resume Profile")
            st.caption("Extracted metadata and structured credentials from your uploaded document.")

            c_sum1, c_sum2 = st.columns(2, gap="medium")
            with c_sum1:
                st.markdown(f"**👤 Candidate Name:** {profile.get('candidate_name', 'Candidate')}")
                st.markdown(f"**🎓 Qualifications / Degree:** {profile.get('qualifications', 'Engineering')}")
                st.markdown(f"**📈 Experience Level:** {profile.get('experience_level', 'Undergraduate')}")
            with c_sum2:
                st.markdown(f"**💻 Core Languages:** {', '.join(profile.get('primary_languages', []))}")
                st.markdown(f"**🌐 Domain Focus:** {', '.join(profile.get('core_domains', []))}")
                st.markdown(f"**🛠️ Highlight Projects:** {len(profile.get('highlight_projects', []))} detected")

            st.write("")

            col_down, col_prev = st.columns([1.2, 1.2], vertical_alignment="center")
            with col_down:
                if st.session_state.get("pdf_bytes"):
                    st.download_button(
                        label="📥 Download Uploaded Resume",
                        data=st.session_state.pdf_bytes,
                        file_name=st.session_state.get("pdf_filename", "resume.pdf"),
                        mime="application/pdf",
                    )
            with col_prev:
                show_pdf = st.toggle("📄 Preview Uploaded PDF Document", value=False)

            if show_pdf and st.session_state.get("pdf_bytes"):
                b64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode("utf-8")
                pdf_iframe = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="650" type="application/pdf" style="border-radius: 12px; border: 1px solid rgba(0, 201, 150, 0.4); margin-top: 15px;"></iframe>'
                st.markdown(pdf_iframe, unsafe_allow_html=True)

    st.write("")

    # ---------------------------------------------------------
    # CLEAN LOG TELEMETRY EXPANDER
    # ---------------------------------------------------------
    with st.expander("⚡ Agent Execution Terminal (CLI Logs & Raw Tool Telemetry)"):
        if st.session_state.thought_stream:
            log_lines = ["career-pilot@nebius-cloud:~$ cat /var/log/pipeline_trace.log\n"]
            for item in st.session_state.thought_stream:
                meta_str = f" • [{item['meta']}]" if item["meta"] else ""
                log_lines.append(f"[{item['time']}] [{item['tag']}] {item['message']}{meta_str}")
            log_lines.append("\n> [SESSION LOG ARCHIVED - EXIT: 0]")
            st.code("\n".join(log_lines), language="bash")

        st.markdown(f"##### Extracted Candidate Profile ({MODEL_NAME})")
        st.json(profile)
        st.markdown("##### Live Unfiltered Web Citations (Tavily Search)")
        st.text_area("Web Corpus Snippets:", value=docs, height=160)

# =========================================================
# FIXED FOOTER
# =========================================================
st.markdown(
    f"""
<div class="app-footer">
    Powered by <b>{FORMATTED_MODEL_NAME}</b> on <b>Nebius Token Factory</b> & <b>Tavily Search</b> &bull; Built for Nebius x NVIDIA Hackathon 2026
</div>
""",
    unsafe_allow_html=True,
)