import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Use a Hugging Face model endpoint. Ensure you have a valid token and endpoint.
HF_API_TOKEN = ("HF_API_TOKEN")
PREDICTIVE_MODEL_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
GENERATIVE_MODEL_URL = "https://api-inference.huggingface.co/models/bigscience/T0_3B"

def predictive_text(prompt: str, max_length=50):
    """
    Calls a Hugging Face model to continue 'prompt' with advanced LLM generation.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(PREDICTIVE_MODEL_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0]["generated_text"]
    return ""

def generative_synonyms(word: str, context: str = ""):
    """
    Leverages an LLM to produce creative or domain-specific synonyms for 'word'
    based on optional 'context'.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Provide synonyms or related expressions for the word '{word}' in the context: '{context}'. "
        "They should be distinct from each other and explained briefly."
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 60,
            "temperature": 0.7
        }
    }
    response = requests.post(GENERATIVE_MODEL_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0]["generated_text"]
    return ""
