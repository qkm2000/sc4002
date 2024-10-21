from nltk import sent_tokenize, word_tokenize
import nltk


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


def process_data(data: list):
    for i in range(len(data)):
        for sent in sent_tokenize(data[i]):
            data[i] = word_tokenize(
                sent,
                language='english',
                preserve_line=True
            )
    return data
