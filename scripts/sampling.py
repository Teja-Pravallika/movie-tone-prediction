from imports import *

rare_tone_indices = [2, 3, 7, 11, 13, 17, 20, 26]

idx_to_tone = {
    0: 'alarming', 1: 'bittersweet', 2: 'comforting', 3: 'empowering',
    4: 'enchanting', 5: 'euphoric', 6: 'fearless', 7: 'futuristic',
    8: 'heartwarming', 9: 'humorous', 10: 'imaginative', 11: 'innovative',
    12: 'inspiring', 13: 'intellectual', 14: 'intriguing', 15: 'investigative',
    16: 'melancholic', 17: 'methodical', 18: 'nostalgic', 19: 'ominous',
    20: 'optimistic', 21: 'perilous', 22: 'profound', 23: 'provocative',
    24: 'raw', 25: 'romantic', 26: 'sophisticated', 27: 'tense', 28: 'wholesome'
}

rare_tone_names = [idx_to_tone[idx] for idx in rare_tone_indices]
print(f"Rare Tone Names (Based on AUC): {rare_tone_names}")

def sample_with_replacement(dataframe, tones_column, rare_tones, sample_size=2):   
    rare_rows = dataframe[dataframe[tones_column].apply(
        lambda tones: any(tone in tones.split(', ') for tone in rare_tones)
    )]
    
    print(f"Number of rows with rare tones: {len(rare_rows)}")
    
    if len(rare_rows) > 0:
        sampled_rows = rare_rows.sample(n=len(rare_rows) * sample_size, replace=True, random_state=42)
        return sampled_rows
    else:
        print("No rows with rare tones found.")
        return pd.DataFrame(columns=dataframe.columns)  

sampled_data = sample_with_replacement(movies_df, 'emotional_tones', rare_tone_names, sample_size=3)
augmented_movies_df = pd.concat([movies_df, sampled_data], ignore_index=True)

print(f"Original Dataset Size: {len(movies_df)}")
print(f"Augmented Dataset Size: {len(augmented_movies_df)}")
augmented_movies_df.to_csv('movies_with_sampling_1.tsv', sep='\t', index=False)
print("Augmented dataset saved as 'movies_with_sampling_1.tsv'")
