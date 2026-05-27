# arXivFlow 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Static Badge](https://img.shields.io/badge/pypi-0.3.0-blue)](https://pypi.org/project/arxivflow/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange.svg)](https://ollama.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-blue.svg)](https://ai.google.dev/gemini-api/docs)
[![arXiv](https://img.shields.io/badge/arXiv-API-red.svg)](https://arxiv.org/help/api/index)

**arXivFlow** is a powerful Python-based automation tool designed to streamline the research paper discovery and tracking process. It autonomously fetches metadata from arXiv, performs AI-driven analysis using **Ollama** or the **Gemini API**, and synchronizes the results with **Google Sheets** and local databases.

---

## ✨ Features

- **Asynchronous API**: Fully rewritten with `asyncio` for high-performance paper retrieval and PDF processing.
- **Automated Retrieval**: Fetch the latest papers from specific arXiv categories (e.g., `cs.AI`, `cs.LG`, `hep-ph`) within any date range.
- **AI Analysis Options**: Uses **Ollama models** for local/private extraction or **Gemini models** for cloud-backed extraction of keywords and contact information (emails/affiliations).
- **Intelligent PDF Handling**: Automatically downloads PDFs and extracts text for deep analysis. Supports custom storage paths and atomic PDF writes.
- **Robust arXiv Requests**: Built-in compliance with arXiv's API guidelines (3-second request intervals), paged metadata retrieval, 429 cooldown handling, retry backoff, and duplicate-result cleanup.
- **Multi-Format Export**: Save your research data to **CSV**, **JSON**, **Excel**, or **SQLite** for flexible offline analysis.
- **Google Sheets Sync**: Seamlessly push compiled research data to a shared Google Sheet for team collaboration.
- **Type-Safe & Modular**: Clean, documented Python code with full type hinting and a class-based architecture.

---

## 🛠️ Prerequisites

1. **Python 3.13+**: Ensure you have a modern Python environment.
2. **Choose an AI backend**:
   - For Ollama, install [Ollama](https://ollama.ai/) and download the required model (e.g., Llama 3.2):
     ```bash
     ollama pull llama3.2
     ```
   - For Gemini, create a Gemini API key and either pass it as `gemini_api_key` or set it as `GOOGLE_AI_API`.
3. **Google Cloud Credentials** for Google Sheets sync:
   - Enable the **Google Sheets** and **Google Drive** APIs.
   - Create a **Service Account** and download the JSON key as `credentials.json`.
   - Ensure the service account has 'Editor' permissions on the sheet.

---

## 🚀 Installation

### From PyPI (Recommended)
```bash
pip install arxivflow
```

### From Source (For Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zjzhao/arXivFlow.git
   cd arXivFlow
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .
   source bin/activate  # On Windows: Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

---

## 📖 Usage

### Quick Start (Async)

```python
import asyncio
import datetime
from arxivflow import arXivFlow

async def main():
    # 1. Initialize the flow with Ollama
    flow = arXivFlow(
        categories=["cs.AI", "cs.CV"], 
        ollama_model="llama3.2",
        max_results=20,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
        request_timeout=60.0
    )

    # 2. Fetch data & Extract info (Keywords/Contacts)
    df = await flow.get_arxiv_data(download_pdfs=True)

    # 3. Save to your preferred formats
    flow.save_to_csv("my_research.csv")
    flow.save_to_sqlite("research.db")

    # 4. Sync with Google Sheets
    flow.save_to_google_sheet(
        sheet_id="YOUR_SHEET_ID", 
        credentials_file="credentials.json"
    )
    
    # 5. Close the client
    await flow.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Gemini Backend

```python
import asyncio
import datetime
import os
from arxivflow import arXivFlow

async def main():
    flow = arXivFlow(
        categories=["cs.AI", "cs.CV"],
        gemini_model="gemini-2.5-flash",
        gemini_api_key=os.getenv("GOOGLE_AI_API"),
        max_results=20,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
    )

    df = await flow.get_arxiv_data(download_pdfs=True)
    flow.save_to_csv("my_research.csv")
    await flow.close()

if __name__ == "__main__":
    asyncio.run(main())
```

If both `ollama_model` and `gemini_model` are provided, Ollama takes precedence. When `gemini_model` is set, a Gemini API key is required; pass `gemini_api_key` directly or set the `GOOGLE_AI_API` environment variable.

---

## 🧱 Request Stability

arXiv can occasionally return slow responses, rate limits, or temporary service errors. arXivFlow now makes the request path more stable by:

- Fetching arXiv metadata in smaller pages instead of relying on one large request.
- Fetching metadata for all requested categories before starting PDF downloads, which avoids PDF download bursts interfering with the next category query.
- Serializing arXiv requests and preserving the recommended 3-second interval.
- Retrying transient failures (`429`, `500`, `502`, `503`, `504`, timeouts, and network errors) with exponential backoff and jitter.
- Applying a longer cooldown after `429` rate-limit responses before making the next arXiv request.
- Respecting `Retry-After` headers when arXiv provides them.
- Using a default 60-second request timeout, configurable with `request_timeout`.
- Writing PDFs to temporary `.part` files first, then atomically replacing the final file only after validating PDF-like content.
- Deduplicating merged output by `arXiv ID`.

For especially large date ranges, prefer smaller `max_results` values or narrower date windows. arXivFlow will page requests internally, but smaller slices are still easier for arXiv and more reliable in practice.

---

## 🏗️ Architecture

The project follows a modular structure for easy extension:

- `src/arxivflow/arxivflow.py`: The main orchestrator class (`arXivFlow`).
- `src/arxivflow/ollama_functions.py`: Local LLM interface using the Ollama API.
- `src/arxivflow/gemini_functions.py`: Gemini API interface for cloud-backed keyword and contact extraction.
- `src/arxivflow/arxiv_functions.py`: Asynchronous arXiv API interaction layer, including paging, rate limiting, retries, and PDF downloads.
- `src/arxivflow/categories.py`: arXiv category definitions.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
