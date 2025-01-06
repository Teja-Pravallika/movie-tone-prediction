from imports import *

movies_df = pd.read_csv('/content/generated_emotional_tones.tsv', delimiter='\t')

movies_df = movies_df.rename(columns = {'Emotional_tones': 'emotional_tones'})

movies_df.shape

movies_df.head()

movies_df.describe()

movies_df['overview'] = movies_df['overview'].replace('No overview found.', np.nan)

print(movies_df.isna().sum())

movies_df = movies_df.dropna(subset = ['overview'])

print()

print(movies_df.isna().sum())

print(movies_df.shape)

invalid_phrases = [
    "I'm sorry",
    "It looks like",
    "It seems",
    "Please provide",
    "Sure!"
]

movies_df = movies_df[~movies_df['emotional_tones'].str.contains('|'.join(invalid_phrases), na=False)]

valid_tones = ['humorous', 'inspiring', 'heartwarming', 'bittersweet', 'euphoric', 'melancholic', 'tense', 'romantic',
               'nostalgic', 'intriguing', 'comforting', 'provocative', 'empowering', 'profound', 'enchanting', 'alarming', 'perilous', 'ominous',
               'fearless', 'imaginative', 'methodical', 'investigative', 'intellectual', 'sophisticated', 'innovative', 'futuristic', 'wholesome', 'raw', 'optimistic']


def filter_tones(tones):
    return ", ".join(tone.strip() for tone in tones.split(", ") if tone.strip() in valid_tones)


movies_df['emotional_tones'] = movies_df['emotional_tones'].apply(filter_tones)

unique_cleaned_tones = sorted(
    set(tone for tones in movies_df['emotional_tones'].str.split(", ") for tone in tones if tone.strip())
)

print(unique_cleaned_tones)

def clean_tones(tones):
    valid = [tone for tone in tones.split(", ") if tone.strip() in valid_tones]
    return ", ".join(valid)

movies_df['emotional_tones'] = movies_df['emotional_tones'].apply(clean_tones)

movies_df['emotional_tones'] = movies_df['emotional_tones'].str.strip()
movies_df = movies_df[movies_df['emotional_tones'] != '']

tones = ['alarming', 'bittersweet', 'comforting', 'empowering', 'enchanting', 'euphoric', 'fearless', 'futuristic', 'heartwarming', 'humorous', 'imaginative', 'innovative', 'inspiring', 'intellectual', 'intriguing', 'investigative', 'melancholic', 'methodical', 'nostalgic', 'ominous', 'optimistic', 'perilous', 'profound', 'provocative', 'raw', 'romantic', 'sophisticated', 'tense', 'wholesome']

tone_to_num = {}
for idx, tone in enumerate(tones):
    tone_to_num[tone] = idx
print(tone_to_num)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

def token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=True))

augmented_movies_df = pd.read_csv('path_to_augmented_movies.csv')

augmented_movies_df['token_count'] = augmented_movies_df['overview'].apply(lambda x: token_count(x, tokenizer))

augmented_movies_df = augmented_movies_df[augmented_movies_df['token_count'] >= 50]

augmented_movies_df = augmented_movies_df.drop(columns=['token_count'])

print("Final Dataset Shape:", augmented_movies_df.shape)
