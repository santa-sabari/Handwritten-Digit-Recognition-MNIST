import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Reshape the data to fit the model input
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32') / 255
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
train_label = to_categorical(train_label, num_classes=10)
test_label = to_categorical(test_label, num_classes=10)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_label, epochs=5, batch_size=32)

# Save the model in the recommended format
model.save('model.keras')
