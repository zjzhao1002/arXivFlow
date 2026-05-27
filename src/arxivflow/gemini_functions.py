import json
from google import genai
from pydantic import BaseModel
from typing import List, Dict, Any

class KeywordsResponse(BaseModel):
    keywords: List[str]

class ContactResponse(BaseModel):
    emails: List[str]
    affiliations: List[str]

class GeminiFunctions:
    def __init__(self, model_name: str, gemini_api_key: str) -> None:
        if not gemini_api_key:
            raise ValueError("Gemini API key is required when gemini_model is set. Pass gemini_api_key or set GOOGLE_AI_API.")
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key
        self.client = genai.Client(api_key=gemini_api_key)
        self._gemini_model_checker()

    def _gemini_model_checker(self) -> bool:
        """
        Checks if the input model_name is a valid Gemini model.
        If it is not valid, sets the model to 'gemini-2.5-flash'.
        """
        try:
            available_models = [m.name for m in self.client.models.list()]
        except Exception as e:
            raise RuntimeError(f"Error listing Gemini models: {e}") from e

        # Check for direct match or models/ prefix match
        if self.model_name in available_models or f"models/{self.model_name}" in available_models:
            return True

        self.model_name = "gemini-2.5-flash"
        return False
    
    async def extract_keywords_gemini(self, title: str, abstract: str) -> List[str]:
        """
        This function extracts keywords from the title and abstract of an arXiv paper.
        Args:
            title: A string of the paper's title.
            abstract: A string of the paper's abstract.
        Returns:
            keywords: A list containing up to 5 keywords.
        """
        if not title or not abstract:
            return []

        print(f"Extracting keywords using {self.model_name} for title: {title}")
        prompt = f"""
        Extract 5 keywords from the following title and abstract:\n\n
        Title: {title}\n\n
        Abstract: {abstract}\n\n
        Respond in JSON format: {{\"keywords\": [\"kw1\", \"kw2\", ...]}}
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": KeywordsResponse,
                }
            )
            raw_content = response.text
            content_json = json.loads(raw_content) # type: ignore
            keywords = content_json.get("keywords", [])
        except json.JSONDecodeError as e:
            print(f"Error during Gemini keywords extraction: {e}")
            keywords = []
        return keywords
    
    async def extract_contact_gemini(self, text: str) -> Dict[str, Any]:
        """
        This function extracts contact information from an arXiv paper.
        Args:
            text: A string extracted from the PDF file. It should contain the contact information.
        Returns:
            content_json: A dictionary that contains the emails and affiliations of authors.
        """
        if not text:
            return {"emails": [], "affiliations": []}

        prompt = f"""
        Extract the emails and affiliations from the following text:\n\n
        {text}\n\n
        Returns the contact information in JSON format: {{\"emails\": [], \"affiliations\": []}}. 
        Affiliations should typically begin with words like "University of", "Institute of", "Department of", etc. 
        Combine department and university names into one full string.
        Don't add any keys to the JSON object. Don't guess if you don't see any contact information in the text.
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ContactResponse,
                    "temperature": 0.0,
                }
            )
            raw_content = response.text
            content_json = json.loads(raw_content) # type: ignore
            return content_json
        except json.JSONDecodeError as e:
            print(f"Error during Gemini contact extraction: {e}")
            return {"emails": [], "affiliations": []}
