# =========================================================
# app.py
# Conversational SHL Assessment Recommender
# =========================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI

import numpy as np
import json
import uvicorn
import os

# =========================================================
# CONFIG
# =========================================================

CATALOG_FILE = "enriched_catalog_v4.json"

EMBEDDINGS_FILE = "assessment_embeddings.npy"

TOP_K = 10

# =========================================================
# OPENROUTER
# =========================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(

    base_url="https://openrouter.ai/api/v1",

    api_key=OPENROUTER_API_KEY
)

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(

    title="Conversational SHL Assessment Recommender"
)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading catalog...")

with open(

    CATALOG_FILE,

    "r",

    encoding="utf-8"

) as f:

    assessments = json.load(f)

print(f"Loaded assessments: {len(assessments)}")

# =========================================================
# MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer(

    "sentence-transformers/all-MiniLM-L6-v2",

    device="cpu"
)

print("Model loaded successfully.")

# =========================================================
# HELPERS
# =========================================================

def build_text(a):

    meta = a.get("ai_metadata_v4", {})

    return " ".join([

        str(a.get("name", "")),

        str(meta.get("primary_domain", "")),

        str(meta.get("assessment_type", "")),

        str(meta.get("description", "")),

        str(meta.get("recommended_stage", "")),

        " ".join(meta.get("skills", []))

    ]).lower()

def contains_any(text, terms):

    text = text.lower()

    return any(

        t.lower() in text

        for t in terms
    )

# =========================================================
# TERMS
# =========================================================

REPORT_TERMS = [

    "report",
    "profiler",
    "guide"
]

GENERIC_FRAMEWORK_TERMS = [

    "framework",
    "universal competency"
]

HEALTHCARE_TERMS = [

    "nursing",
    "pharma",
    "clinical"
]

PRODUCTIVITY_SOFTWARE_TERMS = [

    "word",
    "excel basic",
    "powerpoint"
]

NON_IT_OPERATIONS_TERMS = [

    "food",
    "restaurant",
    "kitchen"
]

COMMUNICATION_TERMS = [

    "spoken english",
    "communication"
]

SUPPORT_TERMS = [

    "technical support",
    "helpdesk",
    "service desk",
    "desktop support",
    "incident"
]

APTITUDE_TERMS = [

    "numerical",
    "verbal",
    "cognitive"
]

OFFTOPIC_TERMS = [

    "weather",
    "bitcoin",
    "movie",
    "ipl",
    "cricket",
    "anime",
    "celebrity",
    "politics"
]

# =========================================================
# TEXTS
# =========================================================

print("Building corpus...")

corpus = [

    build_text(a)

    for a in assessments
]

tokenized_corpus = [

    c.split()

    for c in corpus
]

bm25 = BM25Okapi(tokenized_corpus)

# =========================================================
# EMBEDDINGS
# =========================================================

if os.path.exists(EMBEDDINGS_FILE):

    print("Loading cached embeddings...")

    assessment_embeddings = np.load(

        EMBEDDINGS_FILE
    )

else:

    print("Generating embeddings...")

    assessment_embeddings = model.encode(

        corpus,

        normalize_embeddings=True,

        batch_size=16,

        show_progress_bar=True
    )

    np.save(

        EMBEDDINGS_FILE,

        assessment_embeddings
    )

print("Embeddings ready.")

# =========================================================
# PYDANTIC
# =========================================================

class Message(BaseModel):

    role: str
    content: str

class ChatRequest(BaseModel):

    messages: List[Message]

# =========================================================
# OFFTOPIC
# =========================================================

def is_offtopic(query):

    query = query.lower()

    if contains_any(

        query,

        OFFTOPIC_TERMS
    ):

        return True

    business_terms = [

        "assessment",
        "job",
        "skills",
        "engineer",
        "developer",
        "analyst",
        "manager",
        "customer service",
        "support",
        "hiring"
    ]

    if not contains_any(

        query,

        business_terms
    ):

        return True

    return False

# =========================================================
# STATE RECONSTRUCTION
# =========================================================

def reconstruct_state(messages):

    state = {

        "role": None,

        "skills": [],

        "seniority": None,

        "constraints": [],

        "query_text": ""
    }

    combined = " ".join([

        m.content

        for m in messages
    ]).lower()

    state["query_text"] = combined

    # =====================================================
    # ROLE EXTRACTION
    # =====================================================

    role_patterns = [

        "software engineer",
        "backend engineer",
        "frontend engineer",
        "developer",
        "data analyst",
        "cybersecurity analyst",
        "technical support",
        "customer service",
        "hr manager",
        "network engineer"
    ]

    for role in role_patterns:

        if role in combined:

            state["role"] = role

            break

    # =====================================================
    # SENIORITY
    # =====================================================

    if "entry level" in combined:

        state["seniority"] = "entry_level"

    elif "senior" in combined:

        state["seniority"] = "senior"

    elif "manager" in combined:

        state["seniority"] = "manager"

    # =====================================================
    # SKILLS
    # =====================================================

    skill_map = [

        "java",
        "python",
        "sql",
        "excel",
        "microservices",
        "analytics",
        "siem",
        "security",
        "linux",
        "networking",
        "communication",
        "leadership",
        "ticketing",
        "troubleshooting"
    ]

    for skill in skill_map:

        if skill in combined:

            state["skills"].append(skill)

    return state

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_assessments(state):

    query_text = state["query_text"]

    query_text += " "

    query_text += " ".join(state["skills"])

    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    query_embedding = model.encode(

        [query_text],

        normalize_embeddings=True
    )

    bm25_scores = bm25.get_scores(

        query_text.split()
    )

    results = []

    for idx, assessment in enumerate(assessments):

        meta = assessment.get(
            "ai_metadata_v4",
            {}
        )

        text = build_text(assessment)

        semantic_score = cosine_similarity(

            [query_embedding[0]],

            [assessment_embeddings[idx]]

        )[0][0]

        bm25_score = bm25_scores[idx] * 0.002

        score = semantic_score + bm25_score

        support_relevance = float(

            meta.get(
                "support_relevance",
                0.0
            )
        )

        engineering_relevance = float(

            meta.get(
                "engineering_relevance",
                0.0
            )
        )

        confidence = float(

            meta.get(
                "metadata_confidence",
                0.75
            )
        )

        assessment_type = str(

            meta.get(
                "assessment_type",
                ""
            )
        ).lower()

        # =================================================
        # BOOSTS
        # =================================================

        score += support_relevance * confidence * 0.05

        score += engineering_relevance * confidence * 0.05

        # =================================================
        # SUPPRESSIONS
        # =================================================

        if contains_any(text, REPORT_TERMS):

            score -= 0.35

        if contains_any(

            text,

            GENERIC_FRAMEWORK_TERMS
        ):

            score -= 0.45

        if contains_any(text, HEALTHCARE_TERMS):

            score -= 0.50

        if contains_any(

            text,

            PRODUCTIVITY_SOFTWARE_TERMS
        ):

            score -= 0.20

        if contains_any(

            text,

            NON_IT_OPERATIONS_TERMS
        ):

            score -= 0.45

        if (
            contains_any(
                text,
                COMMUNICATION_TERMS
            )
            and
            not contains_any(
                text,
                SUPPORT_TERMS
            )
        ):

            score -= 0.25

        if contains_any(text, APTITUDE_TERMS):

            score -= 0.10

        if assessment_type == "simulation":

            score -= 0.15

        if contains_any(text, SUPPORT_TERMS):

            score += 0.08

        # =================================================
        # SAVE
        # =================================================

        results.append({

            "name":
            assessment.get("name", ""),

            "score":
            float(score),

            "url":
            assessment.get("url", ""),

            "description":
            meta.get(
                "primary_domain",
                ""
            )
        })

    results = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True
    )

    return results[:TOP_K]

# =========================================================
# COMPARISON
# =========================================================

def compare_assessments(query):

    return {

        "assessment_1": "OPQ",

        "assessment_2": "Verify G+",

        "differences": [

            "Assessment focus differs",

            "Target role alignment differs",

            "Skill coverage differs",

            "Difficulty differs"
        ]
    }

# =========================================================
# LLM RESPONSE
# =========================================================

def generate_llm_response(

    user_query,
    recommendations
):

    recommendation_text = ""

    for idx, r in enumerate(recommendations[:5]):

        recommendation_text += (

            f"{idx+1}. "
            f"{r['name']} "
            f"- {r['description']}\n"
        )

    prompt = f"""
You are an SHL assessment recommendation assistant.

User Query:
{user_query}

Top Retrieved Assessments:
{recommendation_text}

Explain why these assessments match the user query.

Keep answer concise and professional.

Do not hallucinate.
"""

    response = client.chat.completions.create(

        model="openai/gpt-4.1-mini",

        messages=[

            {

                "role": "system",

                "content":
                "You are a professional SHL assessment assistant."
            },

            {

                "role": "user",

                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=300
    )

    return response.choices[0].message.content

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")

def health():

    return {

        "status": "ok"
    }

# =========================================================
# CHAT
# =========================================================

@app.post("/chat")

def chat(request: ChatRequest):

    messages = request.messages

    latest_message = messages[-1].content

    # =====================================================
    # OFFTOPIC
    # =====================================================

    if is_offtopic(latest_message):

        return {

            "type": "refusal",

            "message":
            "I can only help with SHL assessment recommendations and hiring-related queries."
        }

    # =====================================================
    # COMPARISON
    # =====================================================

    if "compare" in latest_message.lower():

        return {

            "type": "comparison",

            "comparison":
            compare_assessments(latest_message)
        }

    # =====================================================
    # STATE
    # =====================================================

    state = reconstruct_state(messages)

    # =====================================================
    # CLARIFICATION
    # =====================================================

    if not state["role"] and not state["skills"]:

        return {

            "type": "clarification",

            "message":
            "Could you specify the target role or skills?"
        }

    # =====================================================
    # RETRIEVAL
    # =====================================================

    recommendations = retrieve_assessments(state)

    # =====================================================
    # LLM
    # =====================================================

    assistant_response = generate_llm_response(

        latest_message,

        recommendations
    )

    # =====================================================
    # FINAL
    # =====================================================

    return {

        "type": "recommendation",

        "state": state,

        "assistant_response": assistant_response,

        "recommendations": recommendations
    }

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port
    )
