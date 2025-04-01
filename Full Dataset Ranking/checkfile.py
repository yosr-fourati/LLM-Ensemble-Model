import pandas as pd

# Load CSV file
df = pd.read_csv("part2_dataset_ranked_responses.csv")

# Identify rows where prompts are not repeated exactly 4 times consecutively
incorrect_rows = []
i = 0
while i < len(df):
    count = 1
    while i + count < len(df) and df.loc[i, "Prompt"] == df.loc[i + count, "Prompt"]:
        count += 1
    if count != 4:
        incorrect_rows.extend(range(i, i + count))
    i += count

# Report incorrect rows
if incorrect_rows:
    print(f"Rows with incorrect repetition count: {incorrect_rows}")
else:
    print("All prompts are correctly repeated 4 times consecutively.")
