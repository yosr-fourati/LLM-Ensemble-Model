import requests
import json
import time
import pandas as pd
from tqdm import tqdm

# Set OpenRouter API Key
API_KEY = ""  # Replace with your actual key
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Read the prompt file
with open("", "r", encoding="utf-8") as file: # Replace with prompts csv file
    prompts = [line.strip() for line in file if line.strip()]  # Remove empty lines

# Define models to test
models = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "perplexity/sonar"
]

# Store responses
responses = []

for index, prompt in tqdm(enumerate(prompts), total=len(prompts)):
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7  # Adjust as needed
        }

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response_json = response.json()

            # Extract response text safely
            generated_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response")

            responses.append({
                "prompt": prompt,
                "model": model,
                "response": generated_text
            })

            time.sleep(1)  # Prevent rate limiting

        except Exception as e:
            print(f"Error with model {model} on prompt {index}: {e}")
            responses.append({
                "prompt": prompt,
                "model": model,
                "response": f"Error: {e}"
            })

# Convert results to a DataFrame and save
output_df = pd.DataFrame(responses)
output_df.to_csv("model_responses.csv", index=False)

print("Finished processing prompts! Responses saved in 'model_responses.csv'.")