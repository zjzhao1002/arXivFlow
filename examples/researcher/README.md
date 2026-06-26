# Example: Outreach Emails as a Researcher

This example shows how a researcher can use `arxivflow` to discover recent
papers related to their own work and draft outreach emails to the authors.

The workflow:

1. Defines a short description of the user's article, abstract, and research
   keywords.
2. Fetches recent `hep-ph` papers from the last 7 days.
3. Uses `arxivflow` keyword extraction to score each paper against the user's
   keywords.
4. Drafts an email for each relevant paper, highlighting the target paper,
   summarizing the user's work, and politely suggesting citation by `llama3.2` model.
5. Writes matching paper metadata, relevance scores, and generated email text
   to `output.csv`.

## Prerequisites

- Python 3.13+
- `arxivflow` installed from PyPI or from the repository root
- [Ollama](https://ollama.ai/) running locally
- The `llama3.2` model available in Ollama

Install the package and model:

```bash
pip install arxivflow
ollama pull llama3.2
```

If you are working from a local checkout, install the package in editable mode
from the repository root instead:

```bash
pip install -e .
```

## Run

From this directory:

```bash
python outreach_emails.py
```

The script creates `output.csv` with these columns:

- `arXiv ID`
- `Authors`
- `Emails`
- `Title`
- `Relevant Score`
- `Outreach Emails`

Only papers with a relevance score greater than `0.0` are included. If no
matching arXiv papers are found for the selected date range, the script still
creates `output.csv` with the same columns and no rows.

## Customize

Edit `outreach_emails.py` to change:

- `user_keywords`, `user_article`, and `user_abstract`
- `categories`, `max_results`, `start_date`, and `end_date` in the `arXivFlow`
  constructor
- the relevance scoring rule in `score_article()`
- the Ollama model name
- the email tone and citation request in `compose_email()`
- the output filename
