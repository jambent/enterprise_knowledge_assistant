# streaming response
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv()

url = os.getenv("LLAMA_URL")
payload = {
    "model": "llama3.2",
    "prompt": "Hello.",
    "stream": True}

with requests.post(url, json=payload, stream=True) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                print(data["response"], end="", flush=True)
            if data.get("done"):
                break