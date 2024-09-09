from dotenv import load_dotenv
from openai import AzureOpenAI
import os

load_dotenv()

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI()

settings = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "tell a joke",
        },
    ], **settings)

# result = completion.to_json()
print(response.choices[0].message.content)