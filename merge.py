import pandas as pd
import glob

files = glob.glob("../data/*.csv")

df_list = []

for file in files:
    print("Reading:", file)

    try:
        df = pd.read_csv(file, low_memory=False)

        # Clean data
        df.columns = df.columns.str.strip()
        df = df.replace([float('inf'), -float('inf')], 0)
        df = df.dropna()

        # Take small sample from each file
        df = df.sample(1000, random_state=42)

        df_list.append(df)

    except Exception as e:
        print("Error:", e)

# Merge
final_df = pd.concat(df_list, ignore_index=True)

# Shuffle
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save
final_df.to_csv("../data/raw.csv", index=False)

print("✅ raw.csv created!")