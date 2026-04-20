# models/deepseek_model.py

from urllib import response

import requests
import os

from transformers import data


class DeepSeekModel:

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.base_url = "https://openrouter.ai/api/v1"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt):

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
            {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, headers=self.headers, json=payload)

        data = response.json()

        if "error" in data:
            raise Exception(f"API Error: {data['error']}")

        if "choices" not in data:
            raise Exception(f"Invalid response format: {data}")

        return data["choices"][0]["message"]["content"]
    

    def embed_text(self, text):
        """
        OpenRouter embeddings endpoint
        """

        url = f"{self.base_url}/embeddings"

        payload = {
            "model": "text-embedding-3-small",  # safer default
            "input": text
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")

        return response.json()["data"][0]["embedding"]