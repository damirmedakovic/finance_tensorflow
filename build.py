
from model import LSTM_model
import pandas as pd
import random
import numpy as np
from sklearn import preprocessing
import time



def remove_headers(ticker):


    with open("Stocks/{}.us.txt".format(ticker), 'r') as f:
        data = f.read().splitlines(True)
    with open("Stocks/{}.us.txt".format(ticker), 'w') as file:
        file.writelines(data[1:])


def classify(current, future):

    if future > current:
        return 1
    else:
        return 0


def build_dataframes(ticker, prediction_period, sequence_length, remove_headers=False):

    if remove_headers:
        remove_headers(ticker)

    df = pd.read_csv("crypto_data/ETH-USD.csv", names=["date", "open", "high", "low", "close", "volume", "openInt"])
    #df = pd.read_csv("Stocks/{}.us.txt".format(ticker), names=["date", "open", "high", "low", "close", "volume", "openInt"])

    df.drop(["open", "high", "low", "openInt"], axis=1, inplace=True)
    df.set_index("date", inplace=True)
    df["future"] = df["close"].shift(-prediction_period)
    labels = list(map(classify, df["close"], df["future"]))
    df["buy/sell"] = labels
    df.drop("future", 1, inplace=True)

    training_data, validation_data = split(df)
    training_features, training_labels = preprocess(training_data, sequence_length)
    validation_features, validation_labels = preprocess(validation_data, sequence_length)

    return training_features, training_labels, validation_features, validation_labels


def split(dataframe):


    split = int(len(dataframe)/(100)*90)
    validation_set = dataframe[split:]
    dataframe = dataframe[:split]
    return dataframe, validation_set


def normalize_data(df):

    df["close"] = df["close"].pct_change()
    df["volume"] = df["volume"].pct_change()
    df["close"] = preprocessing.scale(df["close"].values)
    df["volume"] = preprocessing.scale(df["close"].values)
    df.dropna(inplace=True)

    return df


def preprocess(df, sequence_length):

    df = normalize_data(df)

    sequences = []
    sequence = []

    for val in df.values:
        sequence.append(val[:-1])
        if len(sequence) == sequence_length:
            sequences.append(sequence + [val[-1]])
            sequence = []

    random.shuffle(sequences)

    buys = 0
    sells = 0
    for label in sequences:

        if label[-1] == 0.0:
            sells += 1
        elif label[-1] == 1.0:
            buys += 1

    if buys > sells:
        equalizer = buys - sells
        for seq in range(len(sequences)):
            if equalizer == 0:
                break
            if sequences[seq][-1] == 1.0:
                sequences.pop(seq)
                equalizer -= 1

    if sells > buys:
        equalizer = sells - buys
        for seq in range(len(sequences)):
            if equalizer == 0:
                break
            if sequences[seq][-1] == 0.0:
                sequences.pop(seq)
                equalizer -= 1

    df.dropna(inplace=True)
    random.shuffle(sequences)
    features = []
    labels = []
    for el in sequences:
        features.append(el[:-1])
        labels.append(el[-1])

    features = np.array(features)

    return features, labels


if __name__ == "__main__":


    training_features, training_labels, validation_features, validation_labels = build_dataframes("bgy", 3, 30 )

    LSTM_model(training_features, training_labels, validation_features, validation_labels)
