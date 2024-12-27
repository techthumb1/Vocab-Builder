import os
import json

FEEDBACK_FILE = "user_feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_feedback(feedback_data):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2)

def record_like(word: str, synonym: str):
    """
    Increment a like counter for a given (word, synonym) pair.
    """
    data = load_feedback()
    key = f"{word.lower()}::{synonym.lower()}"
    data[key] = data.get(key, 0) + 1
    save_feedback(data)

def get_likes(word: str, synonym: str):
    data = load_feedback()
    key = f"{word.lower()}::{synonym.lower()}"
    return data.get(key, 0)
