import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from app.features.recommendation.llm.utils import normalize_prompt
from groq import Groq
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-pro",
    ]
else:
    GEMINI_MODELS = []

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

def call_gemini(prompt: str) -> str:
    safe_prompt = normalize_prompt(prompt)

    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(safe_prompt)
            return response.text
        except ResourceExhausted:
            continue
        except Exception:
            continue

    raise RuntimeError("Gemini unavailable")

def call_groq(prompt: str) -> str:
    safe_prompt = normalize_prompt(prompt)

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a senior digital marketing analyst."
            },
            {
                "role": "user",
                "content": safe_prompt
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return completion.choices[0].message.content

def call_llm(prompt: str) -> str:
    try:
        print("[LLM] Trying Gemini...")
        return call_gemini(prompt)
    except Exception as e:
        print(f"[LLM] Gemini failed: {e}")

    try:
        print("[LLM] Trying Groq...")
        return call_groq(prompt)
    except Exception as e:
        print(f"[LLM] Groq failed: {e}")

    print("[LLM] Falling back to rule-based summary")

    return (
        "LLM explanation is currently unavailable. "
        "Based on the aggregated campaign performance, failures are likely "
        "due to weak ROI, inefficient CPA, or poor conversion optimization."
    )
