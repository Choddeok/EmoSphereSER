import pandas as pd

# Load Original label_consensus.csv file provided with dataset
df = pd.read_csv("/workspace/choddeok/hd0/dataset/IS_2025_SER/Labels/labels_consensus_updated.csv")

# Define the emotions
emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
octs = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

# Create a dictionary for one-hot encoding of emotions
one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}

# Filter out rows with undefined EmoClass
df = df[df['EmoClass'].isin(emotion_codes)]

# Apply one-hot encoding for emotions
for i, e in enumerate(emotion_codes):
    df[emotions[i]] = df['EmoClass'].apply(lambda x: one_hot_dict[x][i])

# Normalize EmoAct, EmoVal, EmoDom to [-1, 1]
df['norm_EmoAct'] = 2 * (df['EmoAct'] - 1) / 6 - 1  # Normalize to [-1, 1]
df['norm_EmoVal'] = 2 * (df['EmoVal'] - 1) / 6 - 1  # Normalize to [-1, 1]
df['norm_EmoDom'] = 2 * (df['EmoDom'] - 1) / 6 - 1  # Normalize to [-1, 1]

# Assign octant labels based on normalized values
def assign_octant(row):
    norm_EmoAct, norm_EmoVal, norm_EmoDom = row['norm_EmoAct'], row['norm_EmoVal'], row['norm_EmoDom']
    if norm_EmoAct >= 0 and norm_EmoVal >= 0 and norm_EmoDom >= 0:
        return "I"
    elif norm_EmoAct < 0 and norm_EmoVal >= 0 and norm_EmoDom >= 0:
        return "II"
    elif norm_EmoAct < 0 and norm_EmoVal < 0 and norm_EmoDom >= 0:
        return "III"
    elif norm_EmoAct >= 0 and norm_EmoVal < 0 and norm_EmoDom >= 0:
        return "IV"
    elif norm_EmoAct >= 0 and norm_EmoVal >= 0 and norm_EmoDom < 0:
        return "V"
    elif norm_EmoAct < 0 and norm_EmoVal >= 0 and norm_EmoDom < 0:
        return "VI"
    elif norm_EmoAct < 0 and norm_EmoVal < 0 and norm_EmoDom < 0:
        return "VII"
    elif norm_EmoAct >= 0 and norm_EmoVal < 0 and norm_EmoDom < 0:
        return "VIII"

df['Octant'] = df.apply(assign_octant, axis=1)

# One-hot encode Octant labels
one_hot_octant = {oct_label: [1.0 if oct_label == o else 0.0 for o in octs] for oct_label in octs}
for i, oct_label in enumerate(octs):
    df[oct_label] = df['Octant'].apply(lambda x: one_hot_octant[x][i])

# Select relevant columns for the new CSV
df_final = df[['FileName', *emotions, *octs, 'EmoAct', 'EmoVal', 'EmoDom', 'Split_Set']]

# Save the processed data to a new CSV file
df_final.to_csv('processed_balance_labels_dim_octants.csv', index=False)

print("Processing complete. New file saved as 'processed_labels_dim_octants.csv'")
