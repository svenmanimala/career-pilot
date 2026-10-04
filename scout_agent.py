import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv(override=True)

NEBIUS_KEY = os.environ.get("NEBIUS_API_KEY", "").strip()
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
MODEL_NAME = "nvidia/Nemotron-3-Ultra-550b-a55b"

if not NEBIUS_KEY:
    raise ValueError("Missing NEBIUS_API_KEY in environment or .env file.")
if not TAVILY_KEY:
    raise ValueError("Missing TAVILY_API_KEY in environment or .env file.")

nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_KEY,
)
tavily_client = TavilyClient(api_key=TAVILY_KEY)


def clean_json_string(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


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
  "candidate_name": "Full Name",
  "qualifications": "Degree or background",
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
        if not data.get("primary_languages"):
            data["primary_languages"] = ["General Software / Python"]
        if not data.get("core_domains"):
            data["core_domains"] = ["Software Engineering & Computer Science"]
        if not data.get("highlight_projects"):
            data["highlight_projects"] = ["Foundational coursework & applied software exploration"]
        if not data.get("experience_level"):
            data["experience_level"] = "Undergraduate / Learner"

        data["is_valid"] = True
        return data
    except Exception:
        return {
            "is_valid": True,
            "candidate_name": "Candidate",
            "qualifications": "Computer Science Studies",
            "primary_languages": ["Python", "General Programming"],
            "core_domains": ["Software Development"],
            "highlight_projects": [cleaned[:80] + "..."],
            "experience_level": "Undergraduate",
        }


def discover_live_opportunities(target_query: str) -> dict:
    """Uses Tavily to query current open roles and locate the official career portal."""
    query = f"{target_query} undergraduate research assistant internship openings requirements"
    results = tavily_client.search(query=query, search_depth="advanced", max_results=4)
    hits = results.get("results", [])

    chunks = [
        f"Source ({res.get('url', 'Web')}):\n{res.get('content', '')}"
        for res in hits
    ]
    if not chunks:
        raise RuntimeError(f"No opportunity postings found for: {target_query}")

    careers_url = ""
    for hit in hits:
        u = hit.get("url", "")
        u_lower = u.lower()
        if any(term in u_lower for term in ["career", "job", "opening", "join", "apply", "lab", "team", "people"]):
            careers_url = u
            break

    if not careers_url and hits:
        careers_url = hits[0].get("url", "")

    return {
        "docs": "\n\n".join(chunks),
        "careers_url": careers_url,
    }


def audit_fit_and_generate_pitch(student_profile: dict, opportunity_context: str) -> dict:
    system_prompt = """You are an elite technical career auditor powered by NVIDIA Nemotron.
Cross-examine the student's profile against the live external opportunity/lab requirements.
1. Assign an objective Match Score (0 to 100).
2. Highlight 2 concrete strengths where the student's projects match the lab's real work.
3. Identify 2 specific technical skill gaps or missing technologies they should study.
4. Draft a hyper-personalized, polite, and technically precise cold outreach email citing specific topics from the opportunity.

Return ONLY a valid JSON object matching this schema:
{
  "match_score": 85,
  "match_verdict": "1-sentence executive match verdict",
  "strengths": ["Strength 1", "Strength 2"],
  "skill_gaps": ["Gap 1", "Gap 2"],
  "targeted_cold_pitch": {
    "subject": "Professional email subject line",
    "email_body": "Full body of the cold email"
  },
  "preparation_advice": "Actionable 1-sentence tip on what to build next to close the gap"
}"""

    payload = f"""STUDENT PROFILE:
{json.dumps(student_profile, indent=2)}

TARGET OPPORTUNITY & LAB DATA (FROM LIVE WEB):
{opportunity_context}"""

    response = nebius_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        temperature=0.2,
    )
    return json.loads(clean_json_string(response.choices[0].message.content))