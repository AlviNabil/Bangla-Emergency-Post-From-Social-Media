import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from model_store import dataset_file

df = pd.read_csv(dataset_file("data/train.csv"))
X_train = df.content

vocab_size = 25000
max_length = 100
max_length_2 = 600

trunc_type = "pre"
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)


def tokenize(text):
    text_sequences = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_sequences, maxlen=max_length)
    return text_padded


def tokenize2(text):
    text_sequences = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_sequences, maxlen=max_length_2)
    return text_padded


text_transformer = TfidfVectorizer(
    ngram_range=(1, 3), lowercase=True, max_features=10000
)
text_transformer.fit(X_train)


def transform(text):
    return text_transformer.transform(text)
