# Emotional Tone Prediction In Movies Using LLMs
An LLM-based approach to classify emotional tones in movie descriptions
Introduction
This project explores the use of machine learning and deep learning, leveraging large language models (LLMs) to classify emotional tones in movies based on their descriptions. Using the GPT-4o-mini API, emotional tone labels were generated for movie descriptions, and a fine-tuned RoBERTa model—a deep learning architecture—was trained on the labeled data to perform multi-label classification. To address the challenges of class imbalance, rare tones were identified using precision-recall (PR) curves, and the dataset was augmented through sampling with replacement. This deep learning approach significantly improved the model's precision, recall, and F1-scores, showcasing the effectiveness of handling class imbalance in multi-label classification tasks.

**Features**
* **Label Generation:** Used GPT-4o-mini API to generate up to four emotional tone labels per movie.
* **Deep Learning Model:** Fine-tuned a pre-trained RoBERTa model using PyTorch for tone classification.
* **Class Imbalance Handling:** Identified rare tones using PR curve AUC scores and applied sampling with replacement.
* **Improved Metrics:** Enhanced performance metrics (precision, recall, F1-score) through dataset augmentation.

**Emotional Tones**
The project classifies movies into the following 29 emotional tones:
Humorous, Inspiring, Heartwarming, Bittersweet, Euphoric, Melancholic, Tense, Romantic, Nostalgic, Intriguing, Comforting, Provocative, Empowering, Profound, Enchanting, Alarming, Perilous, Ominous, Fearless, Imaginative, Methodical, Investigative, Intellectual, Sophisticated, Innovative, Futuristic, Wholesome, Raw, Optimistic.
