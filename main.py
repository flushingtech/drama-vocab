from youtube_transcript_api import YouTubeTranscriptApi
import spacy
import nltk  # Natural Language Toolkit

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, brown, wordnet
from nltk.util import ngrams

import pprint
import string
from collections import Counter

video_id = 'TllxLP8sP_w'


# Load English model
# If this doesn't work, run in terminal: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('brown')
nltk.download('wordnet')

ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id)

transcript = ''

for snippet in fetched_transcript:
    transcript += f' {str(snippet.text)}'

transcript = transcript.strip()

doc = nlp(transcript)

# Filter out named entities (e.g., PERSON, ORG, GPE)
filtered_tokens = [token.text.lower() for token in doc 
                   if not token.ent_type_ and token.is_alpha and not token.is_stop]

filler_words = {"wow", "okay", "yeah", "uh", "um", "uh-huh", "oh", "hmm"}
filtered_tokens_without_fillers = [
    word for word in filtered_tokens if word not in filler_words]

filtered_transcript = ' '.join(filtered_tokens_without_fillers)

# Sentence and word tokenization
sentences = sent_tokenize(filtered_transcript)
tokens = [word_tokenize(sent.lower()) for sent in sentences]


# Remove punctuation and stopwords
stop_words = set(stopwords.words('english') + list(string.punctuation))
cleaned_tokens = [[word for word in sent if word not in stop_words] for sent in tokens]
flat_tokens = [word for sent in cleaned_tokens for word in sent]


# Build a frequency distribution from a reference corpus
brown_freq = nltk.FreqDist(w.lower() for w in brown.words())
common_words = set([word for word, freq in brown_freq.most_common(10000)])

# Uncommon words = words not in common word list
vocab_candidates = [word for word in flat_tokens if word not in common_words]


# Create bigrams and trigrams from original token list (not just uncommon ones)
bigrams = list(ngrams(flat_tokens, 2))
trigrams = list(ngrams(flat_tokens, 3))

bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

# Filter out low-frequency ngrams
common_bigrams = [bg for bg, freq in bigram_freq.items() if freq > 1]
common_trigrams = [tg for tg, freq in trigram_freq.items() if freq > 1]

print('Vocabulary candidates: Individual words')
print(vocab_candidates, '\n')
print('Vocabulary candidates: Bigrams')
print(common_bigrams, '\n')
print('Vocabulary candidates: Trigrams')
print(common_trigrams, '\n')

# word_frequency = {}

# for snippet in fetched_transcript:
#     if snippet and snippet.text:
#         words = str(snippet.text).split(' ')
#         for word in words:
#             word = word.strip()
#             if word in word_frequency:
#                 word_frequency[word] += 1
#             else:
#                 word_frequency[word] = 1

