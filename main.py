from youtube_transcript_api import YouTubeTranscriptApi
import pprint
import nltk  # Natural Language Toolkit

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, brown, wordnet
from nltk.util import ngrams
import string

from collections import Counter

video_id = 'TllxLP8sP_w'

nltk.download('brown')
nltk.download('wordnet')

ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id)

transcript = ''

for snippet in fetched_transcript:
    transcript += f' {snippet.text}'

transcript = transcript.strip()


# Sentence and word tokenization
sentences = sent_tokenize(transcript)
tokens = [word_tokenize(sent.lower()) for sent in sentences]

# Remove punctuation and stopwords
stop_words = set(stopwords.words('english') + list(string.punctuation))
cleaned_tokens = [[word for word in sent if word not in stop_words] for sent in tokens]
flat_tokens = [word for sent in cleaned_tokens for word in sent]


# Build a frequency distribution from a reference corpus
brown_freq = nltk.FreqDist(w.lower() for w in brown.words())
common_words = set([word for word, freq in brown_freq.most_common(5000)])

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

pprint.pprint(word_frequency)
