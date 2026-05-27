import os
_parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src_dp = os.path.join(_parent_dp, 'src')
import sys
sys.path.insert(0, _src_dp)

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from arxivflow import gemini_functions

class TestGeminiFunctions(unittest.IsolatedAsyncioTestCase):

    def test_missing_api_key_raises(self):
        with self.assertRaises(ValueError):
            gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="")
    
    @patch('google.genai.Client')
    def test_model_checker_valid_direct(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        gf = gemini_functions.GeminiFunctions(model_name="models/gemini-2.5-flash", gemini_api_key="fake-key")
        self.assertEqual(gf.model_name, "models/gemini-2.5-flash")
        
    @patch('google.genai.Client')
    def test_model_checker_valid_prefixed(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        self.assertEqual(gf.model_name, "gemini-2.5-flash")

    @patch('google.genai.Client')
    def test_model_checker_invalid_fallback(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        gf = gemini_functions.GeminiFunctions(model_name="nonexistent-model", gemini_api_key="fake-key")
        self.assertEqual(gf.model_name, "gemini-2.5-flash")

    @patch('google.genai.Client')
    def test_model_checker_api_error_raises(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.models.list.side_effect = RuntimeError("bad api key")

        with self.assertRaises(RuntimeError):
            gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")

    @patch('google.genai.Client')
    async def test_extract_keywords_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        mock_response = MagicMock()
        mock_response.text = '{"keywords": ["quantum", "computing", "physics"]}'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        keywords = await gf.extract_keywords_gemini("Quantum Computing", "A paper about quantum physics.")
        
        self.assertEqual(keywords, ["quantum", "computing", "physics"])
        mock_client.aio.models.generate_content.assert_called_once()
        
    @patch('google.genai.Client')
    async def test_extract_keywords_empty(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        keywords = await gf.extract_keywords_gemini("", "")
        self.assertEqual(keywords, [])

    @patch('google.genai.Client')
    async def test_extract_contact_gemini(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        mock_response = MagicMock()
        mock_response.text = '{"emails": ["john@example.com"], "affiliations": ["University of Example"]}'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        contact = await gf.extract_contact_gemini("John, john@example.com, University of Example")
        
        self.assertEqual(contact, {"emails": ["john@example.com"], "affiliations": ["University of Example"]})
        mock_client.aio.models.generate_content.assert_called_once()

    @patch('google.genai.Client')
    async def test_extract_contact_empty(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        
        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        contact = await gf.extract_contact_gemini("")
        self.assertEqual(contact, {"emails": [], "affiliations": []})

    @patch('google.genai.Client')
    async def test_extract_keywords_api_error_raises(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [mock_model]
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("quota exceeded"))

        gf = gemini_functions.GeminiFunctions(model_name="gemini-2.5-flash", gemini_api_key="fake-key")
        with self.assertRaises(RuntimeError):
            await gf.extract_keywords_gemini("Quantum Computing", "A paper about quantum physics.")
