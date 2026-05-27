import os
_parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_dp = os.path.join(_parent_dp, 'src')
import sys
sys.path.insert(0, _src_dp)

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from arxivflow import arxivflow

@pytest.mark.anyio
@patch('arxivflow.arxivflow.GeminiFunctions')
@patch('arxivflow.arxivflow.download_pdf')
@patch('arxivflow.arxivflow.search_paper')
async def test_get_arxiv_data_with_gemini(mock_search_paper, mock_download_pdf, mock_gemini_functions, monkeypatch):
    """
    Test that arXivFlow correctly calls GeminiFunctions if gemini_model is specified.
    """
    mock_search_paper.return_value = [
        {
            "arXiv ID": "1234.5678",
            "Title": "A Great Paper",
            "Authors": "An Author",
            "Abstract": "This is a great paper.",
            "Categories": "cs.AI",
            "Published Date": "2026-05-26",
            "Updated Date": "2026-05-26",
            "arXiv URL": "https://arxiv.org/abs/1234.5678",
            "PDF URL": "https://arxiv.org/pdf/1234.5678",
        }
    ]
    
    # Mock download_pdf return value
    mock_download_pdf.return_value = "fake_path/1234.5678.pdf"
    
    # Mock GeminiFunctions instance methods
    mock_instance = mock_gemini_functions.return_value
    mock_instance.extract_contact_gemini = AsyncMock(return_value={"emails": ["john@example.com"], "affiliations": ["University of AI"]})
    mock_instance.extract_keywords_gemini = AsyncMock(return_value=["keyword1", "keyword2"])
    
    # Mock _extract_first_page_text method
    monkeypatch.setattr(arxivflow.arXivFlow, "_extract_first_page_text", lambda self, path: "fake extracted text")
    
    arxiv_flow = arxivflow.arXivFlow(
        categories=["cs.AI"],
        gemini_model="gemini-2.5-flash",
        gemini_api_key="fake-key"
    )
    
    df = await arxiv_flow.get_arxiv_data(download_pdfs=True)
    
    # Assertions
    assert len(df) == 1
    assert df.loc[0, "Emails"] == "john@example.com"
    assert df.loc[0, "Affiliations"] == "University of AI"
    assert df.loc[0, "Keywords"] == "keyword1, keyword2"
    
    mock_instance.extract_contact_gemini.assert_called_once_with("fake extracted text")
    mock_instance.extract_keywords_gemini.assert_called_once_with("A Great Paper", "This is a great paper.")
    mock_gemini_functions.assert_called_once_with(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
    
    await arxiv_flow.close()

@pytest.mark.anyio
@patch('arxivflow.arxivflow.OllamaFunctions')
@patch('arxivflow.arxivflow.GeminiFunctions')
@patch('arxivflow.arxivflow.download_pdf')
@patch('arxivflow.arxivflow.search_paper')
async def test_get_arxiv_data_precedence_execution(mock_search_paper, mock_download_pdf, mock_gemini_functions, mock_ollama_functions, monkeypatch):
    """
    Test that if both Ollama and Gemini are set, Ollama takes precedence during get_arxiv_data.
    """
    mock_search_paper.return_value = [
        {
            "arXiv ID": "1234.5678",
            "Title": "A Great Paper",
            "Authors": "An Author",
            "Abstract": "This is a great paper.",
            "Categories": "cs.AI",
            "Published Date": "2026-05-26",
            "Updated Date": "2026-05-26",
            "arXiv URL": "https://arxiv.org/abs/1234.5678",
            "PDF URL": "https://arxiv.org/pdf/1234.5678",
        }
    ]
    
    # Mock download_pdf return value
    mock_download_pdf.return_value = "fake_path/1234.5678.pdf"
    
    # Mock OllamaFunctions instance methods
    mock_ollama_instance = mock_ollama_functions.return_value
    mock_ollama_instance.extract_contact_ollama = MagicMock(return_value={"emails": ["ollama@example.com"], "affiliations": ["University of Ollama"]})
    mock_ollama_instance.extract_keywords_ollama = MagicMock(return_value=["ollama1", "ollama2"])
    
    # Mock _extract_first_page_text method
    monkeypatch.setattr(arxivflow.arXivFlow, "_extract_first_page_text", lambda self, path: "fake extracted text")
    
    arxiv_flow = arxivflow.arXivFlow(
        categories=["cs.AI"],
        ollama_model="llama3.2",
        gemini_model="gemini-2.5-flash",
        gemini_api_key="fake-key"
    )
    
    df = await arxiv_flow.get_arxiv_data(download_pdfs=True)
    
    # Assertions
    assert len(df) == 1
    assert df.loc[0, "Emails"] == "ollama@example.com"
    assert df.loc[0, "Affiliations"] == "University of Ollama"
    assert df.loc[0, "Keywords"] == "ollama1, ollama2"
    
    # GeminiFunctions should NOT be instantiated and NOT called
    mock_gemini_functions.assert_not_called()
    
    await arxiv_flow.close()
