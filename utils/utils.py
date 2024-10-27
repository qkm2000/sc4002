import nltk
from nltk import word_tokenize
import re


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


def process_data(data: list):
    for i in range(len(data)):
        data[i] = clean_text(data[i])
        data[i] = tokenize_text(data[i])
    return data


def clean_text(text, stopwords=None):
    # remove non-ascii characters
    text = text.encode("ascii", "ignore").decode('utf-8', 'ignore')

    # remove URLs, mentions, hashtags
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # case folding
    text = text.lower()

    # split up hyphenated words into separate tokens
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)

    # remove slashes or dots between words - separate the words
    text = re.sub(r'(\w)[/\.](\w)', r'\1 \2', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # normalize whitespaces (multiple whitespaces --> single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text):
    # Word tokenizer
    word_tokens = word_tokenize(text)
    return word_tokens
