# 1. Import required libraries
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# --- Data loading & preparation ---

# Assume the file has been uploaded to the current working directory in Colab
file_path = "datasets/project_c_samples_3k.parquet"

try:
    df = pd.read_parquet(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure it is uploaded to your Colab session.")
    exit()

# --- Expand the data ---
print("Columns:", df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSample rows:")
print(df.head(3).T)

example = df.iloc[0]
print(json.loads(example['names']))
print(json.loads(example['base_addresses']))

# --- Create labels & split the dataset ---

# Create a binary label:
# If 'id' and 'base_id' are the same, treat it as a match (1); otherwise as a non-match (0).
# This is a very simple heuristic label for demonstration and homework purposes.
df['label'] = (df['id'] == df['base_id']).astype(int)

# Show the class balance. This helps students check for class imbalance issues.
print("\nLabel distribution based on (id == base_id):")
print(df['label'].value_counts())

# Split the dataset into train and test sets with stratification to preserve class balance.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("\nDataset split complete:")
print(f"Train size: {train_df.shape}")
print(f"Test size:  {test_df.shape}\n")


# --- Sample visualization helper (no logic changes; comments/docstrings clarified) ---

def visualize_sample(sample, sample_type=""):
    """
    Pretty-print a single DataFrame row to compare the candidate place vs. its base (reference) place.

    Parameters
    ----------
    sample : pd.Series
        One row from the DataFrame (e.g., df.iloc[0]).
    sample_type : str
        A label describing why this sample was selected (e.g., "High similarity (same name)").
    """

    def safe_json_loads(data, key):
        """
        Attempt to parse a JSON-like field and extract a human-readable string.

        Expected input formats in this dataset:
        - A JSON string containing either a list of dicts or a dict.
        - Already-parsed Python dict/list objects.
        - Occasionally None or other types.

        Behavior:
        - If it's a list with at least one element, try to show the 'freeform' field of the first element.
          (Many place records store freeform text like addresses or names here.)
        - If it's a dict, return data.get(key, "...") where `key` is typically 'primary' or 'freeform'.
        - Otherwise, cast to string.

        This keeps the display compact and readable for students.
        """
        if data is None:
            return "N/A"
        try:
            parsed_data = json.loads(data) if isinstance(data, str) else data
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                # If it's a list, show the 'freeform' field of the first item when available.
                return parsed_data[0].get('freeform', parsed_data[0])
            elif isinstance(parsed_data, dict):
                # If it's a dict, try the requested key (e.g., 'primary' or 'freeform').
                return parsed_data.get(key, "Not applicable or missing info")
            else:
                # Fallback for numbers, strings, etc.
                return str(parsed_data)
        except (json.JSONDecodeError, TypeError, KeyError):
            # If parsing fails, just return raw data as string.
            return str(data)

    print("=" * 80)
    print(f"Sample type: {sample_type}\n")

    print(f"{'Attribute':<15} | {'Candidate Place':<60} | {'Base Place':<60}")
    print("-" * 140)

    print(f"{'ID':<15} | {sample['id']:<60} | {sample['base_id']:<60}")
    print(f"{'Name':<15} | {safe_json_loads(sample['names'], 'primary'):<60} | {safe_json_loads(sample['base_names'], 'primary'):<60}")
    print(f"{'Category':<15} | {safe_json_loads(sample['categories'], 'primary'):<60} | {safe_json_loads(sample['base_categories'], 'primary'):<60}")
    print(f"{'Address':<15} | {safe_json_loads(sample['addresses'], 'freeform'):<60} | {safe_json_loads(sample['base_addresses'], 'freeform'):<60}")

    print("=" * 80 + "\n")


# --- Sample selection logic (demonstrates positive/negative examples for students) ---

# 1) Find a "highly similar/matching" example:
#    Criterion: the JSON strings of the primary names are exactly the same.
#    This is a strong match signal for demonstration purposes.
similar_name_candidates = df[df['names'] == df['base_names']]

if not similar_name_candidates.empty:
    similar_sample = similar_name_candidates.iloc[0]
    visualize_sample(similar_sample, sample_type="Highly similar / potential match (names identical)")
else:
    print("No samples found in the dataset where the primary names are exactly identical.")

# 2) Find a "non-match" example:
#    Criterion: the JSON strings of the primary names differ.
#    This is a strong non-match signal for demonstration purposes.
dissimilar_name_candidates = df[df['names'] != df['base_names']]

if not dissimilar_name_candidates.empty:
    dissimilar_sample = dissimilar_name_candidates.iloc[0]
    visualize_sample(dissimilar_sample, sample_type="Non-match (names differ)")
else:
    print("No samples found in the dataset where the primary names are different.")