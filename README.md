# Conversational SHL Assessment Recommender

An AI-powered conversational recommendation system for SHL assessments using Hybrid Retrieval, RAG-style orchestration, semantic search, BM25 ranking, and LLM-based response generation.

---

# Overview

This project implements a conversational API that:

- Recommends relevant SHL assessments
- Asks clarification questions
- Handles dynamic user constraints
- Compares assessments
- Uses semantic retrieval + BM25 hybrid ranking
- Generates grounded conversational responses using LLMs

The system is designed as a production-style retrieval pipeline rather than a simple keyword matcher.

---

# Features

## Conversational Recommendations

Supports natural language queries such as:

- "Recommend assessments for backend software engineer with Java and microservices experience"
- "Suggest assessments for customer service roles"
- "Recommend tests for cybersecurity analyst"

---

## Clarification Questions

If insufficient information is provided, the system asks follow-up questions.

Example:

```json
{
  "type": "clarification",
  "message": "Could you specify the target role or required skills?"
}
