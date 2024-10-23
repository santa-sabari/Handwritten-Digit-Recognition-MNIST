import numpy as np
import cv2
from tensorflow.keras.models import load_model  # Use Keras directly
from collections import deque

def load_model_from_file():
    # Load the model from the .keras file
    model = load_model("model.keras")
    print("Loaded model from disk")
    return model

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smaller kernel for less blur
    
    # Use adaptive thresholding instead of global thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def get_contours(thresh):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    # Load the trained model
    loaded_model = load_model_from_file()

    # Open the video capture
    cap = cv2.VideoCapture(0)

    # Buffer for storing predictions
    prediction_buffer = deque(maxlen=5)  # Store the last 5 predictions
    predicted_digit = ''

    # Define coordinates for the green box (x, y, width, height)
    box_x, box_y, box_w, box_h = 0, 0, 300, 300  # Adjust as necessary

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        # Draw the green box on the original frame
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) inside the box
        roi = img[box_y:box_y + box_h, box_x:box_x + box_w]
        thresh = preprocess_image(roi)
        contours = get_contours(thresh)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:  # Adjust this threshold as necessary
                x, y, w, h = cv2.boundingRect(contour)
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))  # Resize to 28x28
                newImage = np.array(newImage)
                newImage = newImage.flatten()  # Flatten the image
                newImage = newImage.reshape(1, 28, 28, 1)  # Reshape for the model

                # Normalize the input
                newImage = newImage.astype('float32') / 255.0  # Scale to [0, 1]

                # Make prediction
                prediction = loaded_model.predict(newImage)
                predicted_digit_current = str(np.argmax(prediction[0]))  # Get the predicted digit
                
                # Append the current prediction to the buffer
                prediction_buffer.append(predicted_digit_current)

                # Update the predicted_digit only if the buffer is full
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    # Count occurrences of each predicted digit
                    predicted_digit = max(set(prediction_buffer), key=prediction_buffer.count)

        # Display the predicted digit
        cv2.putText(img, "Predicted Digit: " + predicted_digit, (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the original image with the bounding box
        cv2.imshow("Frame", img)
        cv2.imshow("Thresholded Image", thresh)  # Optional: Show the thresholded image for debugging

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
