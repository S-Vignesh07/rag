---
title: InnovateCorp Onboarding Assistant
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# InnovateCorp Onboarding Assistant

A RAG chatbot (LangChain + Gemini + FAISS) that answers onboarding questions from a
company KT guide, with a Gradio chat UI.

## Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space, SDK = **Gradio**.
2. Upload `app.py` and `requirements.txt` from this folder.
3. In the Space settings, add a secret named `GOOGLE_API_KEY` with your Gemini API key.
4. The Space will build and launch automatically.

## Run locally / on any server

    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-key-here"
    python app.py

Then open the printed local URL in your browser.
