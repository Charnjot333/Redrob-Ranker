# Redrob Candidate Ranker

AI-powered candidate ranking system for the Redrob Hackathon.

## Problem
Rank 100,000 candidates against a Senior AI Engineer JD and return top 100.

## Pipeline
1. Rule-based scoring (title, location, notice period, availability)
2. Skills scoring (proficiency + duration + endorsements + trust multiplier)
3. Career analysis (GitHub activity, reliability, stability, experience)
4. Semantic embeddings (BAAI/bge-small-en-v1.5)
5. Hybrid ranking (weighted combination of all scores)
6. LLM re-ranking (Groq/Llama 3.1 on top 200 candidates)

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce submission
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Sandbox
[Live Demo]([your-streamlit-url-here](https://redrob-ranker-gkmudfzuxibjvf6x3mpgxa.streamlit.app/))

## AI Tools Used
- Claude for architecture discussion and code review
- Groq/Llama 3.1 for LLM re-ranking
