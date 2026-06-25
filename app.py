import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from groq import Groq

# page config
st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker")
st.write("Rank candidates against the Senior AI Engineer JD using full AI pipeline")

# load groq key from secrets first
groq_key = os.environ.get("GROQ_API_KEY", "")
if groq_key:
    st.success("Groq API key loaded successfully.")
else:
    groq_key = st.text_input("Enter your Groq API key", type="password")

# JD config - exactly from job_description.docx
jd = {
    "consulting_firms": [
        "TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
        "Capgemini", "Tech Mahindra", "HCL", "Mindtree"
    ],
    "preferred_locations": ["Pune", "Noida"],
    "acceptable_locations": ["Hyderabad", "Mumbai", "Delhi", "Gurugram", "Gurgaon"],
    "notice_period_ideal_days": 30,
    "notice_period_acceptable_days": 60,
}

jd_text = """
Senior AI Engineer founding team role at an AI hiring company.
Looking for someone with 6-8 years experience, 4-5 years in applied ML/AI
at product companies. Must have experience building production search,
ranking, or recommendation systems.
Skills needed: Python, embeddings, vector databases like Pinecone,
Weaviate, Qdrant, FAISS, Elasticsearch.
Experience with NLP, semantic search, learning to rank, NDCG, MRR, MAP.
Should have shipped real systems to real users at meaningful scale.
Not a fit: pure research background, only consulting company experience,
only recent LangChain tutorials, stopped coding 18+ months ago.
Preferred location: Pune or Noida, India.
"""

# JD important skills - from job description
jd_important_skills = [
    "python", "embeddings", "faiss", "elasticsearch", "opensearch",
    "pinecone", "weaviate", "qdrant", "vector", "nlp",
    "recommendation", "ranking", "search", "pytorch", "tensorflow",
    "spark", "kafka", "aws", "gcp", "docker", "kubernetes"
]

proficiency_weight = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.00
}

# ── helper functions ──────────────────────────────────────────────────────────

def classify_title(title):
    t = title.lower()
    core = [
        "machine learning", "ml engineer", "ai engineer", "data scientist",
        "nlp", "recommendation", "applied scientist", "deep learning",
        "search engineer"
    ]
    adjacent = [
        "software engineer", "backend engineer", "data engineer",
        "full stack", "platform engineer", "research scientist",
        "devops", "cloud engineer", "site reliability"
    ]
    if any(k in t for k in core):
        return "core_ai_ml"
    elif any(k in t for k in adjacent):
        return "adjacent_tech"
    return "unrelated"


def compute_availability_score(signals):
    try:
        last_active = datetime.strptime(
            signals["last_active_date"], "%Y-%m-%d"
        ).date()
        days_inactive = (date.today() - last_active).days
        recency = max(0, 1 - days_inactive / 180)
    except Exception:
        recency = 0.5
    response_rate = signals.get("recruiter_response_rate", 0.0)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.3
    completeness = signals.get("profile_completeness_score", 0) / 100
    verified = sum([
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False)
    ]) / 3
    return round(
        0.30 * recency + 0.30 * response_rate +
        0.15 * open_to_work + 0.15 * completeness +
        0.10 * verified, 3
    )


def extract_features(candidate, jd):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    history = candidate.get("career_history", [])

    title_tier = classify_title(profile["current_title"])
    consulting_only = bool(history) and all(
        job["company"] in jd["consulting_firms"] for job in history
    )

    location = profile.get("location", "")
    country = profile.get("country", "")
    willing = signals.get("willing_to_relocate", False)

    if any(p.lower() in location.lower() for p in jd["preferred_locations"]):
        location_fit = "preferred"
    elif any(a.lower() in location.lower() for a in jd["acceptable_locations"]):
        location_fit = "acceptable"
    elif country.lower() == "india" and willing:
        location_fit = "relocatable"
    else:
        location_fit = "poor"

    notice_days = signals.get("notice_period_days", 999)
    if notice_days <= jd["notice_period_ideal_days"]:
        notice_fit = "ideal"
    elif notice_days <= jd["notice_period_acceptable_days"]:
        notice_fit = "acceptable"
    else:
        notice_fit = "high_bar"

    availability = compute_availability_score(signals)

    return {
        "candidate_id": candidate["candidate_id"],
        "title": profile["current_title"],
        "title_tier": title_tier,
        "years_of_experience": profile.get("years_of_experience", 0),
        "location_fit": location_fit,
        "consulting_only": consulting_only,
        "notice_fit": notice_fit,
        "availability_score": availability,
    }


def compute_rule_score(row):
    score = 0
    score += {"core_ai_ml": 40, "adjacent_tech": 20, "unrelated": 0}.get(
        row["title_tier"], 0
    )
    score += {"preferred": 20, "acceptable": 15, "relocatable": 8, "poor": 0}.get(
        row["location_fit"], 0
    )
    score += {"ideal": 15, "acceptable": 8, "high_bar": 0}.get(
        row["notice_fit"], 0
    )
    score += row["availability_score"] * 15
    if row["consulting_only"]:
        score -= 30
    return score


def compute_skills_score(candidate):
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get(
        "redrob_signals", {}
    ).get("skill_assessment_scores", {})

    total_score = 0
    for skill in skills:
        name = skill["name"].lower()
        is_relevant = any(kw in name for kw in jd_important_skills)
        if not is_relevant:
            continue

        proficiency = proficiency_weight.get(skill.get("proficiency"), 0.25)
        duration = min(skill.get("duration_months", 0) / 24, 1.0)
        endorsements = min(skill.get("endorsements", 0) / 10, 1.0)

        duration_months = skill.get("duration_months", 0)
        endorse_count = skill.get("endorsements", 0)

        if duration_months >= 12 or endorse_count >= 5:
            trust = 1.0
        elif duration_months >= 3 or endorse_count >= 1:
            trust = 0.6
        else:
            trust = 0.2

        assessment_bonus = 0
        for assess_name, assess_score in assessment_scores.items():
            if any(kw in assess_name.lower() for kw in jd_important_skills):
                assessment_bonus = (assess_score / 100) * 2

        skill_score = (
            proficiency * 0.4 + duration * 0.3 + endorsements * 0.3
        ) * trust * 10
        total_score += skill_score + assessment_bonus

    return round(total_score, 2)


def compute_github_score(candidate):
    github_score = candidate["redrob_signals"].get("github_activity_score", -1)
    if github_score == -1:
        return 0
    return round((github_score / 100) * 10, 2)


def compute_reliability_score(candidate):
    signals = candidate["redrob_signals"]
    interview_rate = signals.get("interview_completion_rate", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_rate = 0.5
    return round((interview_rate * 0.6 + offer_rate * 0.4) * 10, 2)


def compute_career_stability_score(candidate):
    history = candidate.get("career_history", [])
    if not history:
        return 5
    short_stints = sum(
        1 for job in history if job.get("duration_months", 0) < 18
    )
    short_ratio = short_stints / len(history)
    return round((1 - short_ratio) * 10, 2)


def compute_experience_score(candidate):
    years = candidate["profile"].get("years_of_experience", 0)
    if 6 <= years <= 8:
        return 10
    elif 5 <= years <= 9:
        return 7
    elif 4 <= years <= 10:
        return 4
    else:
        return 1


def build_candidate_text(candidate):
    parts = []
    profile = candidate["profile"]
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(
            f"{job.get('title', '')} at {job.get('company', '')}: "
            f"{job.get('description', '')}"
        )
    skills = [s["name"] for s in candidate.get("skills", [])]
    parts.append("Skills: " + ", ".join(skills))
    edu_parts = []
    for edu in candidate.get("education", []):
        edu_parts.append(
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )
    if edu_parts:
        parts.append("Education: " + ", ".join(edu_parts))
    return " ".join(parts)


def build_candidate_summary(candidate):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    summary = f"""
Candidate ID: {candidate['candidate_id']}
Title: {profile['current_title']}
Years of Experience: {profile['years_of_experience']}
Location: {profile['location']}, {profile['country']}
Headline: {profile.get('headline', '')}
Summary: {profile.get('summary', '')}

Career History:
"""
    for job in candidate.get("career_history", []):
        summary += (
            f"- {job['title']} at {job['company']} "
            f"({job['duration_months']} months): {job['description']}\n"
        )
    skills = [
        f"{s['name']} ({s['proficiency']}, {s['duration_months']}mo)"
        for s in candidate.get("skills", [])
    ]
    summary += f"\nSkills: {', '.join(skills)}"
    summary += f"""

Platform Signals:
- Open to work: {signals['open_to_work_flag']}
- Notice period: {signals['notice_period_days']} days
- Response rate: {signals['recruiter_response_rate']}
- Last active: {signals['last_active_date']}
- GitHub activity: {signals['github_activity_score']}
"""
    return summary


# ── main app ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Step 1: Upload Candidates File")
st.info(
    "Upload a .json or .jsonl file containing candidate profiles. "
    "For demo use sample_candidates.json (50 candidates)"
)

uploaded_file = st.file_uploader(
    "Upload candidates file",
    type=["json", "jsonl"]
)

top_n = st.slider("Number of top candidates to show", 5, 20, 10)

candidates = None

if uploaded_file:
    try:
        filename = uploaded_file.name
        if filename.endswith(".jsonl"):
            candidates = []
            for line in uploaded_file:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        else:
            candidates = json.load(uploaded_file)
            if isinstance(candidates, dict):
                candidates = [candidates]
        st.success(f"Loaded {len(candidates)} candidates successfully")
    except Exception as e:
        st.error(f"Error loading file: {e}")

st.markdown("---")

if candidates and groq_key:
    if st.button("Run Ranking", type="primary"):

        st.subheader("Running Full Pipeline")
        st.write(f"Processing {len(candidates)} candidates...")

        # step 1: extract features + rule score
        with st.spinner("Step 1/5: Extracting features and rule scores..."):
            feature_rows = [extract_features(c, jd) for c in candidates]
            df = pd.DataFrame(feature_rows)
            df["rule_score"] = df.apply(compute_rule_score, axis=1)
            st.write("✓ Rule scores computed")

        # step 2: skills score
        with st.spinner("Step 2/5: Computing skills scores..."):
            df["skills_score"] = [compute_skills_score(c) for c in candidates]
            st.write("✓ Skills scores computed")

        # step 3: career score
        with st.spinner("Step 3/5: Computing career scores..."):
            github_scores = [compute_github_score(c) for c in candidates]
            reliability_scores = [compute_reliability_score(c) for c in candidates]
            stability_scores = [compute_career_stability_score(c) for c in candidates]
            experience_scores = [compute_experience_score(c) for c in candidates]

            df["career_score"] = [
                round(
                    g * 0.30 + r * 0.25 + s * 0.25 + e * 0.20, 2
                )
                for g, r, s, e in zip(
                    github_scores, reliability_scores,
                    stability_scores, experience_scores
                )
            ]
            st.write("✓ Career scores computed")

        # step 4: semantic similarity
        with st.spinner("Step 4/5: Computing semantic similarity (1-2 min)..."):
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            jd_vector = model.encode(jd_text)
            texts = [build_candidate_text(c) for c in candidates]
            vectors = model.encode(
                texts, batch_size=32, show_progress_bar=False
            )
            sim_scores = cosine_similarity(
                vectors, jd_vector.reshape(1, -1)
            ).flatten()
            df["semantic_score"] = (sim_scores * 100).round(2)
            st.write("✓ Semantic scores computed")

        # step 5: combine all 4 scores - exactly like Colab
        with st.spinner("Step 5/5: Computing final scores..."):
            scaler = MinMaxScaler()
            df["rule_norm"] = scaler.fit_transform(df[["rule_score"]])
            df["skills_norm"] = scaler.fit_transform(df[["skills_score"]])
            df["career_norm"] = scaler.fit_transform(df[["career_score"]])
            df["semantic_norm"] = scaler.fit_transform(df[["semantic_score"]])

            # same weights as Colab
            df["final_score"] = (
                df["rule_norm"]     * 0.30 +
                df["skills_norm"]   * 0.20 +
                df["career_norm"]   * 0.20 +
                df["semantic_norm"] * 0.30
            ) * 100
            st.write("✓ Final scores computed")

        # LLM re-ranking - top candidates only
        st.write(f"Running LLM re-ranking on top {top_n * 2} candidates...")
        try:
            client = Groq(api_key=groq_key)
            top_candidates = df.sort_values(
                "final_score", ascending=False
            ).head(top_n * 2).reset_index(drop=True)

            results = []
            progress = st.progress(0)
            total = len(top_candidates)

            for idx, row in top_candidates.iterrows():
                cand = next(
                    c for c in candidates
                    if c["candidate_id"] == row["candidate_id"]
                )
                summary = build_candidate_summary(cand)[:2000]
                prompt = f"""
You are an expert technical recruiter evaluating candidates for this role:

JOB DESCRIPTION:
{jd_text}

CANDIDATE PROFILE:
{summary}

Respond in EXACTLY this format, nothing else:
SCORE: [number between 0 and 100]
REASONING: [1-2 complete sentences explaining why this candidate fits or does not fit]
"""
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=150
                    )
                    text = response.choices[0].message.content.strip()
                    lines = text.split("\n")
                    score_val = float(
                        [l for l in lines if l.startswith("SCORE:")][0]
                        .replace("SCORE:", "").strip()
                    )
                    reasoning = (
                        [l for l in lines if l.startswith("REASONING:")][0]
                        .replace("REASONING:", "").strip()
                    )
                except Exception:
                    score_val = 50.0
                    reasoning = "Could not evaluate"

                results.append({
                    "candidate_id": row["candidate_id"],
                    "title": row["title"],
                    "years_of_experience": row["years_of_experience"],
                    "llm_score": score_val,
                    "reasoning": reasoning
                })
                progress.progress((idx + 1) / total)

            st.write("✓ LLM re-ranking complete")

        except Exception as e:
            st.error(f"LLM error: {e}")
            results = []

        # show results
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(
                "llm_score", ascending=False
            ).head(top_n).reset_index(drop=True)
            results_df["rank"] = range(1, len(results_df) + 1)

            st.markdown("---")
            st.subheader(f"Top {top_n} Candidates")
            st.dataframe(
                results_df[[
                    "rank", "candidate_id", "title",
                    "years_of_experience", "llm_score", "reasoning"
                ]],
                use_container_width=True
            )

            # download button
            submission_df = results_df[[
                "candidate_id", "rank", "llm_score", "reasoning"
            ]].copy()
            submission_df.columns = [
                "candidate_id", "rank", "score", "reasoning"
            ]
            csv = submission_df.to_csv(index=False)
            st.download_button(
                label="Download submission.csv",
                data=csv,
                file_name="submission.csv",
                mime="text/csv"
            )

elif not groq_key:
    st.warning("Please enter your Groq API key above")
elif not candidates:
    st.warning("Please upload a candidates file to continue")