import os
_parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_dp = os.path.join(_parent_dp, 'src')
import sys
sys.path.insert(0, _src_dp)

from unittest.mock import MagicMock, patch

from arxivflow import ollama_functions

def _mock_ollama_models():
    return MagicMock(models=[{"model": "llama3.2:latest"}])

def _mock_chat_response(content):
    return {"message": {"content": content}}

def test_extract_contact_ollama():
    # Test case 1: Valid text with emails and affiliations
    text1 = """
    John Doe
    University of Example
    john.doe@example.com
    """
    with patch("ollama.list", return_value=_mock_ollama_models()), patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_chat_response(
            '{"emails": ["john.doe@example.com"], "affiliations": ["University of Example"]}'
        )
        ollama_func = ollama_functions.OllamaFunctions(model_name="llama3.2")
        result = ollama_func.extract_contact_ollama(text1)
        assert "john.doe@example.com" in result["emails"]
        assert "University of Example" in result["affiliations"]

        # Test case 2: Text with no contact information
        text2 = """
        This is a paper abstract with no contact information.
        """
        mock_chat.return_value = _mock_chat_response('{"emails": [], "affiliations": []}')
        result = ollama_func.extract_contact_ollama(text2)
        assert result["emails"] == []
        assert result["affiliations"] == []

        # Test case 3: Text with multiple emails and affiliations
        text3 = """
        Jane Smith, University of Sample, jane.smith@sample.edu
        Bob Johnson, Sample Institute, bob.johnson@sample.org
        """
        mock_chat.return_value = _mock_chat_response(
            '{"emails": ["jane.smith@sample.edu", "bob.johnson@sample.org"], '
            '"affiliations": ["University of Sample", "Sample Institute"]}'
        )
        result = ollama_func.extract_contact_ollama(text3)
        assert "jane.smith@sample.edu" in result["emails"]
        assert "bob.johnson@sample.org" in result["emails"]
        assert "University of Sample" in result["affiliations"]
        assert "Sample Institute" in result["affiliations"]

def test_extract_keywords_ollama():
    # Test case 1: Valid title and abstract
    title1 = "A Novel Approach to Machine Learning"
    abstract1 = "This paper presents a novel approach to machine learning that outperforms existing methods."
    with patch("ollama.list", return_value=_mock_ollama_models()), patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_chat_response(
            '{"keywords": ["machine learning", "novel approach"]}'
        )
        ollama_func = ollama_functions.OllamaFunctions(model_name="llama3.2")
        keywords1 = ollama_func.extract_keywords_ollama(title1, abstract1)
        assert isinstance(keywords1, list)
        assert len(keywords1) > 0

        # Test case 2: Empty title and abstract
        title2 = ""
        abstract2 = ""
        keywords2 = ollama_func.extract_keywords_ollama(title2, abstract2)
        assert isinstance(keywords2, list)
        assert len(keywords2) == 0

        # Test case 3: Title and abstract with special characters
        title3 = "An Analysis of @#$%^&*() in Data Science"
        abstract3 = "This study analyzes the impact of special characters in data science applications."
        mock_chat.return_value = _mock_chat_response(
            '{"keywords": ["special characters", "data science"]}'
        )
        keywords3 = ollama_func.extract_keywords_ollama(title3, abstract3)
        assert isinstance(keywords3, list)
        assert len(keywords3) > 0
