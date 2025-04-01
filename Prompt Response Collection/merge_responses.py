import pandas as pd
import os

# Define file paths
files = {
    "summaries": r"model_responses_summaries.csv",
    "english_1": r"model_responses_500-English(story) (1).csv",
    "english_2": r"model_responses_FIRST 500_English.csv",
    "math_1": r"ARC_responses-Math.csv",
    "math_2": r"model_responses_OpenThoughts114k- Math.csv",
    "coding_1": r"model_responses_HumanEval- Code.csv",
    "coding_2": r"model_responses_MBPP-Code.csv",
    "coding_3": r"model_responses_python_code_instructions_600.csv",
    "reasoning_1": r"model_responses_BBH ( Reasoning).csv",
    "reasoning_2": r"model_responses_MMLU_Pro( Reasoning).csv",
    "reasoning_3": r"model_responses_BBH_navigate.csv",

}

# Load and merge datasets with source tracking
dfs = []
for source, filepath in files.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["source"] = source  # Add source column
        dfs.append(df)
    else:
        print(f"Warning: File not found - {filepath}")

# Concatenate all datasets into one
df_all = pd.concat(dfs, ignore_index=True)

# Count errors before removal
error_patterns = ["Error: Expecting value:", "HTTPSConnectionPool", "Response ended prematurely"]
errors_df = df_all[df_all["response"].str.contains('|'.join(error_patterns), case=False, na=False)]
error_count = len(errors_df)

# Remove error responses
df_all = df_all[~df_all["response"].str.contains('|'.join(error_patterns), case=False, na=False)]

# Count and extract duplicates before removal
df_all["prompt_count"] = df_all.groupby("prompt")["prompt"].transform("count")
duplicates_df = df_all[df_all["prompt_count"] > 4]

# Keep only the first 4 occurrences per prompt
df_all = df_all.groupby("prompt").head(4)

duplicate_count = len(duplicates_df) - len(df_all)

df_all.drop(columns=["prompt_count"], inplace=True)

# Count final prompts per category
total_prompts_per_category = df_all.groupby("source")["prompt"].nunique()

# Save cleaned dataset
df_all.to_csv(r"merged_cleaned_dataset.csv", index=False)

# Save duplicates and errors separately
duplicates_df.to_csv(r"duplicates.csv", index=False)
errors_df.to_csv(r"errors.csv", index=False)

print(f"Dataset merging and cleaning complete. Saved as 'merged_cleaned_dataset.csv'")
print(f"Total errors removed: {error_count}")
print(f"Total duplicate prompts removed: {duplicate_count}")
print("Final number of unique prompts per category (after removing duplicates and errors):")
print(total_prompts_per_category)
