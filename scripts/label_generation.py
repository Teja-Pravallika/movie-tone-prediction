!pip install openai

!export OPENAI_API_KEY = '# Enter the unique API key here'

import pandas as pd
movies_df = pd.read_csv('/content/movies_metadata.csv', low_memory = False)
movies_df = movies_df[['original_title', 'overview' ]]
print(movies_df.shape)
print(movies_df.isna().sum())
movies_df = movies_df.dropna(subset = ['overview'])
print(movies_df.isna().sum())
movies_df = movies_df.sample(n = 15000, random_state = 42)
print(movies_df.shape)
movies_sample = movies_df.sample(n= 20, random_state = 42)
print(movies_sample.shape)

!pip install tqdm

from openai import OpenAI
from tqdm import tqdm
import pandas as pd

client = OpenAI(api_key='your-api-key')
movies_df = pd.read_csv('/content/movies_with_tone_labels_final.tsv', sep='\t')

Based on the provided movie description, identify up to four dominant emotional tones that best represent the story's essence (minimum one, maximum four). The tones should capture the movie's emotional depth, themes, challenges, relationships, or overall mood.

Stick strictly to the following list of tones: 
[humorous, inspiring, heartwarming, bittersweet, euphoric, melancholic, tense, romantic, nostalgic, intriguing, comforting, provocative, empowering, profound, enchanting, alarming, perilous, ominous, fearless, imaginative, methodical, investigative, intellectual, sophisticated, innovative, futuristic, wholesome, raw, optimistic].

Select tones based on the below guidelines:
1. Humorous: Use for lighthearted or comedic stories with amusing elements.
2. Inspiring: Select for narratives of personal growth, triumph, or motivational arcs.
3. Heartwarming: Apply to stories with emotionally uplifting and fulfilling moments.
4. Bittersweet: Choose for narratives blending joy and sorrow, with happiness intertwined with emotional complexity or loss.
5. Melancholic: Use for reflective or lingering sadness in the story.
6. Raw: Suitable for unfiltered and deeply honest emotional storytelling.
7. Wholesome: Apply to morally uplifting and feel-good narratives that promote positivity.
8. Comforting: Use for stories that are soothing, reassuring, or relaxing.
9. Nostalgic: Choose for tales evoking longing for the past or fond memories.
10. Intellectual: Suitable for thought-provoking and idea-driven storytelling.
11. Sophisticated: Apply to refined, elegant, or artistically complex narratives.
12. Innovative: Use for creative or groundbreaking ideas in storytelling.
13. Futuristic: Suitable for visionary or forward-looking stories, often involving technology.
14. Euphoric: Choose for moments or stories filled with overwhelming joy or exhilaration.
15. Tense: Apply to suspenseful or anxiety-inducing narratives.
16. Romantic: Use for stories focusing on love and relationships.
17. Intriguing: Suitable for mysterious or curiosity-evoking plots.
18. Provocative: Use for bold, thought-challenging stories that push societal norms.
19. Empowering: Choose for tales emphasizing personal growth, independence, or confidence.
20. Enchanting: Apply to magical, captivating, or awe-inspiring narratives.
21. Fearless: Use for bold, audacious, or daring stories.
22. Imaginative: Suitable for highly creative or fantasy-driven tales.
23. Optimistic: Choose for stories that inspire hope and positivity.
24. Alarming: Apply to urgent and escalating crises that evoke fear.
25. Perilous: Use for life-threatening dangers and hazardous environments central to the plot.
26. Ominous: Choose for a sense of dread or foreboding, particularly where bad outcomes are anticipated.
27. Methodical: Suitable for structured, process-driven narratives, like scientific discoveries.
28. Investigative: Apply to mystery-solving or truth-seeking plots.

Only select tones explicitly reflected in the movie description. Your output must be a comma-separated string containing only the tones listed above. Do not include any additional or modified tones.

Description:
"""

new_tags = []
for idx, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Processing Movies"):
    movie_description = row['overview']
    input_text = prompt + movie_description

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are an assistant that generates relevant tones from movie descriptions. Adhere to the rules provided for tone selection."},
                {"role": "user", "content": input_text}
            ]
        )
        new_tags.append(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        new_tags.append("error")
        
    if (idx + 1) % 500 == 0 or idx == len(movies_df) - 1:
        temp_df = movies_df.iloc[:len(new_tags)].copy()
        temp_df['Emotional_tones'] = new_tags
        temp_save_path = f'generated_emotional_tones_partial_{idx + 1}.tsv'
        temp_df.to_csv(temp_save_path, sep='\t', index=False)
        print(f"Intermediate results saved to {temp_save_path}")

movies_df['Emotional_tones'] = new_tags
final_save_path = 'generated_emotional_tones.tsv'
movies_df.to_csv(final_save_path, sep='\t', index=False)
print(f"Final DataFrame has been saved to '{final_save_path}'")
