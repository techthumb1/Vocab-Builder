# Advanced Dictionary–Thesaurus with LLM & Embeddings

## Overview

This application provides classical dictionary/thesaurus functionality using WordNet and advanced NLP features with transformer embeddings and LLMs via Hugging Face Inference API.

## Requirements

- Python 3.8+  
- `pip install -r requirements.txt`  
- Access to a Hugging Face Inference API token if generating synonyms via LLM

## Setup & Configuration

**Clone the repo**:

   ```bash
   git clone https://github.com/yourusername/my-advanced-dictionary-thesaurus.git
   cd my-advanced-dictionary-thesaurus
```

## Run the application

```bash
streamlit run app.py
```

Visit the displayed localhost URL (e.g., http://localhost:8501).

## Usage

- Analyze Word: Provides dictionary definitions, synonyms, advanced synonyms, etc.
- Generate Synonyms: Interacts with the Hugging Face model to create additional synonyms or phrases.

## Contributing

Feel free to submit issues or pull requests.

## License

[MIT](https://choosealicense.com/licenses/mit/)