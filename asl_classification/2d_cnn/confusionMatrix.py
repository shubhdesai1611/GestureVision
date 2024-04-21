import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

model = tf.keras.models.load_model("./2d_cnn_model.h5")

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

# Get predicted labels
y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)

# Convert one-hot encoded labels back to categorical labels
y_test_categorical = np.argmax(y_test_encoded, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_categorical, y_pred)

conf_matrix = confusion_matrix(y_test_categorical, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("ConfusionMatrix.png")
plt.close()
