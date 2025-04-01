import pandas as pd
import requests
import itertools
import time
import logging
from tqdm import tqdm 
import os

# Configure logging to write errors to a file
logging.basicConfig(filename="ranking_errors.log", level=logging.ERROR)

# OpenRouter API details
API_KEY = ""
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "perplexity/sonar"
]

df = pd.read_csv("Full_dataset_with_id.csv")

df = df.groupby("prompt").head(4).reset_index(drop=True)
print(df.head())

# Create a dictionary mapping prompts to their responses and models (normalize responses)
prompts_dict = {}
for _, row in df.iterrows():
    prompt = row["prompt"]
    response_norm = str(row["response"]).strip()
    if prompt not in prompts_dict:
        prompts_dict[prompt] = []
    prompts_dict[prompt].append({
        "model": row["model"], 
        "response": response_norm, 
        "source": row["source"], 
        "ID": row["ID"]
    })

# Function to send a request to OpenRouter
def query_llm(prompt, response1, response2, model):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an unbiased judge evaluating responses."},
            {"role": "user", "content": f"Given the following user prompt:\n{prompt}\n\n Compare these two responses and decide which one is **better in terms of clarity, accuracy, and completeness**:\n\n Response 1: {response1}\nResponse 2: {response2}\n\n Please choose the better response and reply with only 'Response 1' or 'Response 2'."}
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        result = response.json()
        
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logging.error(f"Warning: No valid response from {model} for query.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying {model}: {e}")
        return None

# Prepare output CSV file
output_csv = "ranked_responses_final.csv"
# If file doesn't exist, create it with header
if not os.path.isfile(output_csv):
    header = ["Prompt", "Model", "Source", "ID", "Response"] + MODELS + ["Final Score", "Rank"]
    pd.DataFrame(columns=header).to_csv(output_csv, index=False)

# Calculate total comparisons for progress bar
total_comparisons = sum(len(list(itertools.combinations(resp_list, 2))) * len(MODELS) for resp_list in prompts_dict.values())

with tqdm(total=total_comparisons, desc="Ranking Responses", unit="comparison") as pbar:
    # Process each prompt group separately
    for prompt, response_data in prompts_dict.items():
        # Pre-check: if any response in the prompt is empty, skip the whole prompt
        if any(not item["response"] for item in response_data):
            logging.error(f"Skipping prompt '{prompt}' because one or more responses are empty.")
            continue

        # Create all pairwise groupings first
        response_pairs = list(itertools.combinations(response_data, 2))
        # Initialize scores using normalized responses as keys
        scores = {item["response"]: {model: 0 for model in MODELS} for item in response_data}
        
        prompt_failed = False  # Flag to check if any API call fails for this prompt
        
        # Process each pair of responses
        for pair in response_pairs:
            response1 = pair[0]["response"]
            response2 = pair[1]["response"]
            votes = {
                response1: {model: 0 for model in MODELS},
                response2: {model: 0 for model in MODELS}
            }
            
            for model in MODELS:
                start_time = time.time()  # Track start time
                try:
                    winner = query_llm(prompt, response1, response2, model)
                except Exception as e:
                    logging.error(f"Error querying {model} for prompt '{prompt}': {e}")
                    prompt_failed = True
                    break

                elapsed_time = time.time() - start_time  # Time taken for this request
                
                if winner is None:
                    logging.error(f"Skipping prompt '{prompt}' due to API error for model '{model}'.")
                    prompt_failed = True
                    break
                else:
                    winner_lower = winner.lower()
                    if "response 1" in winner_lower:
                        votes[response1][model] = 1
                    elif "response 2" in winner_lower:
                        votes[response2][model] = 1
                    elif "tie" in winner_lower:
                        votes[response1][model] = 0.5
                        votes[response2][model] = 0.5
                
                pbar.update(1)
                pbar.set_postfix({"Last query time (s)": round(elapsed_time, 2)})
            
            # If any API call failed for this prompt, break out of processing this prompt
            if prompt_failed:
                break

            # Update scores for the current pair based on votes
            for resp in [response1, response2]:
                for model in MODELS:
                    scores[resp][model] += votes[resp][model]
        
        # If there was an error with any API call for this prompt, skip saving and move to the next prompt
        if prompt_failed:
            continue

        # Assign ranks based on the aggregated scores for this prompt
        ranked_results = []
        response_ranks = []
        for response, model_scores in scores.items():
            total_score = sum(model_scores.values())
            response_ranks.append((response, model_scores, total_score))
        
        # Sort responses by final score (descending)
        response_ranks.sort(key=lambda x: -x[2])
        
        # Assign ranks with tie handling
        prev_score, rank = None, 0
        for i, (response, model_scores, total_score) in enumerate(response_ranks):
            if total_score == prev_score:
                assigned_rank = rank
            else:
                assigned_rank = i + 1
                rank = assigned_rank
            prev_score = total_score
            
            try:
                original_entry = next(item for item in response_data if item["response"] == response)
            except StopIteration:
                logging.error(f"Could not find original entry for prompt '{prompt}' and response '{response}'. Skipping this response.")
                continue
            
            ranked_results.append([
                prompt, 
                original_entry["model"], 
                original_entry["source"], 
                original_entry["ID"], 
                response, 
                *[model_scores[m] for m in MODELS], 
                total_score, 
                assigned_rank
            ])
        
        # Save the results for this prompt incrementally (append to CSV)
        try:
            temp_df = pd.DataFrame(
                ranked_results, 
                columns=["Prompt", "Model", "Source", "ID", "Response"] + MODELS + ["Final Score", "Rank"]
            )
            temp_df.to_csv(output_csv, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error saving to CSV for prompt '{prompt}'. Error: {e}")

print("Ranking completed! Results saved incrementally to 'ranked_responses_final.csv'. Check 'ranking_errors.log' for any errors.")
