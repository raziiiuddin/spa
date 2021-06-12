from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from clean_tweets import clean_tweets
from tokenizer import tokenizer

MAX_SEQUENCE_LENGTH = 280
model = load_model("C:\\Users\\raziu\\PycharmProjects\\ProfileAnalysis\\models\\analyzer_2.h5")


def get_labels(tweets):
    cleaned_tweets = clean_tweets(tweets)
    print("----------------------------clean tweets-------------------------------------------")
    print(cleaned_tweets)
    sequences = tokenizer.texts_to_sequences(cleaned_tweets)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = model.predict(data)
    return labels
