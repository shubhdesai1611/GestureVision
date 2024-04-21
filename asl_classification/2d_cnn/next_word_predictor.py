import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataset = []

with open("dataset.txt", "r") as file:
    dataset = file.read()

tokenizer = Tokenizer()

tokenizer.fit_on_texts([dataset])

with open("tokenizer.pickle", "wb") as tokens:
    pickle.dump(tokenizer, tokens, protocol=pickle.HIGHEST_PROTOCOL)

input_sequences = []
for sentence in dataset.split("\n"):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[: i + 1])

max_len = max([len(x) for x in input_sequences])

padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

X = padded_input_sequences[:, :-1]

y = padded_input_sequences[:, -1]

y = to_categorical(y, num_classes=283)

split_ratio = [0.8, 0.1, 0.1]  # 80% training, 10% validation, 10% test
total_samples = len(X)
split_indices = [int(total_samples * ratio) for ratio in split_ratio]
X_train, X_val, X_test = np.split(
    X, [split_indices[0], split_indices[0] + split_indices[1]]
)
y_train, y_val, y_test = np.split(
    y, [split_indices[0], split_indices[0] + split_indices[1]]
)


model = Sequential()
model.add(Embedding(283, 100))
model.add(LSTM(150))
model.add(Dense(283, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

training_history = model.fit(X, y, epochs=100, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


def save_plot(
    data,
    firstKey,
    secondKey,
    title,
    xlabel,
    ylabel,
    firstLabel,
    secondLabel,
    filename,
):
    plt.plot(data.history[firstKey], label=firstLabel)
    plt.plot(data.history[secondKey], label=secondLabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()


save_plot(
    training_history,
    "loss",
    "val_loss",
    "RNN Training and Validation Loss",
    "Epochs",
    "Loss",
    "Training Loss",
    "Validation Loss",
    "RNN Training_and_Validation_Loss.png",
)
save_plot(
    training_history,
    "accuracy",
    "val_accuracy",
    "RNN Training and Validation Accuracy",
    "Epochs",
    "Accuracy",
    "Training Accuracy",
    "Validation Accuracy",
    "RNN Training_and_Validation_Accuracy.png",
)
