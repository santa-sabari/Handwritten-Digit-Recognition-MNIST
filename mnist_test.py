import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

# Main function
def main():
    # Load MNIST dataset
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    # Preprocess the data
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32') / 255
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32') / 255

    # One-hot encoding the labels
    train_label = to_categorical(train_label, num_classes=10)
    test_label = to_categorical(test_label, num_classes=10)

    # Load the model
    try:
        loaded_model = load_model("model.keras")  # Updated to match the new save format
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate the model
    scores = loaded_model.evaluate(test_data, test_label, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Make predictions
    predictions = loaded_model.predict(test_data)

    # Display predictions for the first 10 test images
    for i in range(10):
        plt.imshow(test_data[i].reshape(28, 28), cmap='gray')  # Reshape for proper display
        plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(test_label[i])}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
