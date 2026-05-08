import ollama
import json

class OllamaFunctions:
    def __init__(self, model_name: str) -> None:
        """
        The constructor of the OllamaFunctions class. This class can interact with the local Ollama API.
        Args:
            model_name: The model name of the local LLM. If this model is not found, the program will try to pull it.
        """
        self.model_name = model_name
        if not self._ollama_model_checker():
            try:
                print(f"Model '{self.model_name}' not found in Ollama. Attempting to pull the model...")
                self._ollama_pull_model()
            except Exception as e:
                print(f"Error occurred while pulling the model: {e}")

    def _ollama_model_checker(self) -> bool:
        """
        This function checks if the given model is available locally. 
        It checks for exact matches or matches with tags (like :latest or :3b).
        If a match is found via a tag, it updates self.model_name to the full name.
        Args:
            None
        Returns:
            Returns True if the model is available, False otherwise.
        """
        available_models = ollama.list()
        model_names = [model['model'] for model in available_models.models]
        
        # 1. Exact match (e.g., "llama3.2:latest" or "llama3.2:3b")
        if self.model_name in model_names:
            return True
            
        # 2. If no tag provided, check for common tags
        if ":" not in self.model_name:
            # Check for :latest first
            latest_name = f"{self.model_name}:latest"
            if latest_name in model_names:
                self.model_name = latest_name
                return True
            
            # Check for any other tag (e.g., "llama3.2" matches "llama3.2:3b")
            for name in model_names:
                if name.startswith(f"{self.model_name}:"):
                    self.model_name = name
                    return True
                    
        return False

    def _ollama_pull_model(self) -> None:
        """
        This function pulls the model using Ollama.
        Args:
            None
        Returns:
            None
        """
        ollama.pull(self.model_name)
        print(f"Model '{self.model_name}' pulled successfully.")

    def extract_keywords_ollama(self, title: str, abstract: str) -> list:
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
        response = ollama.chat(
            model=self.model_name, 
            format="json",
            messages=[{"role": "user", "content": prompt}])
        raw_content = response['message']['content']
        try:
            content_json = json.loads(raw_content)
            keywords = content_json.get("keywords", [])
        except json.JSONDecodeError:
            print(f"Error decoding JSON from Ollama response: {raw_content}")
            keywords = []
        return keywords
    
    def extract_contact_ollama(self, text: str) -> dict:
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
        response = ollama.chat(
            model=self.model_name, 
            format="json", 
            messages=[{"role": "user", "content": prompt}], 
            options={"temperature": 0.0}
            )
        raw_content = response['message']['content']
        try:
            content_json = json.loads(raw_content)
            return content_json
        except json.JSONDecodeError:
            print(f"Error decoding JSON from Ollama response: {raw_content}")
            return {"emails": [], "affiliations": []}