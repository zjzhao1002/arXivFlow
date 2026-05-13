# arXivFlow 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Static Badge](https://img.shields.io/badge/pypi-0.2.0-blue)](https://pypi.org/project/arxivflow/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange.svg)](https://ollama.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-API-red.svg)](https://arxiv.org/help/api/index)

**arXivFlow** is a powerful Python-based automation tool designed to streamline the research paper discovery and tracking process. It autonomously fetches metadata from arXiv, performs local AI-driven analysis using **Ollama (e.g., Llama 3.2)**, and synchronizes the results with **Google Sheets** and local databases.

---

## ✨ Features

- **Asynchronous API**: Fully rewritten with `asyncio` for high-performance paper retrieval and PDF processing.
- **Automated Retrieval**: Fetch the latest papers from specific arXiv categories (e.g., `cs.AI`, `cs.LG`, `hep-ph`) within any date range.
- **Local AI Analysis**: Uses **Ollama models (e.g., Llama 3.2)** to extract keywords and contact information (emails/affiliations) directly from PDF text. No cloud API costs or data privacy concerns.
- **Intelligent PDF Handling**: Automatically downloads PDFs and extracts text for deep analysis. Supports custom storage paths.
- **Robust Rate Limiting**: Built-in compliance with arXiv's API guidelines (3-second request intervals).
- **Multi-Format Export**: Save your research data to **CSV**, **JSON**, **Excel**, or **SQLite** for flexible offline analysis.
- **Google Sheets Sync**: Seamlessly push compiled research data to a shared Google Sheet for team collaboration.
- **Type-Safe & Modular**: Clean, documented Python code with full type hinting and a class-based architecture.

---

## 🛠️ Prerequisites

1. **Python 3.13+**: Ensure you have a modern Python environment.
2. **Ollama**: Install [Ollama](https://ollama.ai/) and download the required model (e.g., Llama 3.2):
   ```bash
   ollama pull llama3.2
   ```
3. **Google Cloud Credentials**:
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
    # 1. Initialize the flow
    flow = arXivFlow(
        categories=["cs.AI", "cs.CV"], 
        ollama_model="llama3.2",
        max_results=20,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7)
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

---

## 🏗️ Architecture

The project follows a modular structure for easy extension:

- `src/arxivflow/arxivflow.py`: The main orchestrator class (`arXivFlow`).
- `src/arxivflow/ollama_functions.py`: Local LLM interface using the Ollama API.
- `src/arxivflow/arxiv_functions.py`: Asynchronous arXiv API interaction layer.
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
