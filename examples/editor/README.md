# Example: Outreach Emails for a Scientific Journal

This example shows how a journal editor can use `arxivflow` to find recent
papers on arXiv and draft personalized outreach emails to the authors.

The workflow:

1. Fetches recent `hep-ph` papers from the last 7 days.
2. Downloads and analyzes PDFs with the local Ollama model configured in the
   script.
3. Uses each paper's title, author list, and abstract to draft an invitation
   email for "Example Scientific Journal" by `llama3.2` model.
4. Writes the selected paper metadata and generated email text to `output.csv`.

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
- `Outreach Emails`

If no matching arXiv papers are found for the selected date range, the script
still creates `output.csv` with the same columns and no rows.

## Customize

Edit `outreach_emails.py` to change:

- `categories`, `max_results`, `start_date`, and `end_date` in the `arXivFlow`
  constructor
- the Ollama model name
- the journal name and tone in `compose_email()`
- the output filename
