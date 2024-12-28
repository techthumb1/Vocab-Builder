import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class LLMIntegration:
    def __init__(self):
        # If using HF Inference endpoints:
        self.hf_token = os.getenv("HF_API_TOKEN")
        self.llm_url = os.getenv("LLM_API_URL", "")
        #self.llm_key = os.getenv("LLM_API_KEY", "")

        # Legacy endpoints (if you still want them around, though they may cause 400 errors):
        self.predictive_url = "https://api-inference.huggingface.co/models/bigcode/starcoder"
        #self.generative_url = "https://api-inference.huggingface.co/models/bigscience/T0_3B"
        self.generative_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    def _make_request(self, url: str, prompt: str, max_tokens: int, temp: float = 0.5):
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temp,
                "return_full_text": False
            }
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        return response.json()

    def predictive_text(self, prompt: str, max_length: int = 50) -> str:
        """
        Calls the StarCoder model for next-word / text completion.
        Might fail with 400 if the model is restricted or the prompt is invalid.
        """
        try:
            data = self._make_request(self.predictive_url, prompt, max_length)
            # Some HF endpoints return a list with "generated_text"
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            return ""
        except Exception as e:
            return f"Error in predictive text: {str(e)}"

    def generative_synonyms(self, word: str, context: str = "") -> str:
        """
        Calls the flan-t5-base endpoint to generate synonyms. 
        Might fail with 400 if the model is restricted or the prompt is invalid.
        """
        prompt = f"Provide synonyms or related expressions for '{word}' in context: '{context}'"
        try:
            data = self._make_request(self.generative_url, prompt, 60)
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            return ""
        except Exception as e:
            return f"Error in generative synonyms: {str(e)}"

    def llm_writing_advice(self, word: str, context_sentence: str = "") -> str:
        """
        Calls a custom LLM endpoint set in .env (LLM_API_URL, LLM_API_KEY)
        for more advanced writing advice, synonyms, usage examples, etc.
        """
        if not self.llm_url:
            return "No LLM endpoint configured. Set LLM_API_URL in .env"
            
        prompt = (
            f"You are a highly-skilled and efficient technical writing assistant. Provide an extended discussion of '{word}', "
            f"Including advanced synonyms, usage examples, and writing advice. "
            f"Context: '{context_sentence}'"
            f"2 synonyms or related expressions."
            f"1 rewriting tip to integrate the word in a sentence."
            f"Example usage in a short paragraph."
        )
        
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.5
                }
            }
            response = requests.post(self.llm_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # The structure of 'data' can vary depending on the endpoint
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return "No valid response from LLM."
            
        except Exception as e:
            return f"Error in writing advice: {str(e)}"


def llm_writing_advice(word: str, context_sentence: str = "") -> str:
    """
    Top-level function that the rest of the code can import directly:

    from src.llm_integration import llm_writing_advice

    This instantiates LLMIntegration under the hood to prevent 
    the 'cannot import name' error.
    """
    integrator = LLMIntegration()
    return integrator.llm_writing_advice(word, context_sentence)
