import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt


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


data_dict = pickle.load(open("data_2Dcnn.pickle", "rb"))

# Inspect data
data = data_dict["data"]
labels = np.asarray(data_dict["labels"])

labels = np.asarray([int(label) for label in labels])

lengths = [len(item) for item in data]
desired_length = 42

padded_data = [
    (
        seq + [[0, 0]] * (desired_length - len(seq))
        if len(seq) < desired_length
        else seq[:desired_length]
    )
    for seq in data
]
data = np.array(padded_data)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
)

# One-hot encode the labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
# print("Y train encoded: ", y_train_encoded, y_train_encoded.shape)
# Reshape the data to 3D for Conv1D layer
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

model = Sequential()
model.add(
    Conv2D(filters=64, kernel_size=(2, 2), activation="relu", input_shape=(42, 2, 1))
)
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model
training_history = model.fit(
    X_train_reshaped, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_reshaped, y_test_encoded)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

save_plot(
    training_history,
    "loss",
    "val_loss",
    "Training and Validation Loss",
    "Epochs",
    "Loss",
    "Training Loss",
    "Validation Loss",
    "Training_and_Validation_Loss.png",
)
save_plot(
    training_history,
    "accuracy",
    "val_accuracy",
    "Training and Validation Accuracy",
    "Epochs",
    "Accuracy",
    "Training Accuracy",
    "Validation Accuracy",
    "Training_and_Validation_Accuracy.png",
)

model.save("2d_cnn_model.h5")
