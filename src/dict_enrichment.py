# src/dictionary_enrichment.py

import os
import requests

def fetch_dictionary_data(word: str) -> dict:
    """
    Fetch dictionary data (pronunciation, etymology, definitions, synonyms, examples)
    from The Free Dictionary API or a similar source.
    """
    base_url = os.getenv("FREE_DICTIONARY_API", "https://api.dictionaryapi.dev/api/v2/entries/en")
    url = f"{base_url}/{word}"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        return {}

    data = resp.json()

    # Data structure can vary. The Free Dictionary returns a list of entries.
    if not isinstance(data, list) or len(data) == 0:
        return {}

    # We'll just return the first entry for brevity.
    return data[0]

def parse_pronunciation(data: dict) -> str:
    """Extract phonetic info if available."""
    if "phonetics" in data and data["phonetics"]:
        # Return the first available 'text' field
        for ph in data["phonetics"]:
            if "text" in ph:
                return ph["text"]
    return ""

def parse_etymology(data: dict) -> str:
    """
    The Free Dictionary might store etymology or origin in 'origin', 
    but not always. We'll look for it.
    """
    return data.get("origin", "")

def parse_meanings(data: dict) -> dict:
    """
    Return a structured dict of partOfSpeech -> list of definitions, synonyms, examples
    """
    if "meanings" not in data:
        return {}
    output = {}
    for m in data["meanings"]:
        pos = m.get("partOfSpeech", "unknown")
        definitions = m.get("definitions", [])
        output[pos] = []
        for d in definitions:
            definition_text = d.get("definition", "")
            example_text = d.get("example", "")
            syns = d.get("synonyms", [])
            output[pos].append({
                "definition": definition_text,
                "example": example_text,
                "synonyms": syns
            })
    return output
