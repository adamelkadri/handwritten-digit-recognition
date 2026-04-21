
import os #handles file operations - useful to test images
import cv2 #OpenCV library for image processing
import numpy as np #for numerical operations
import tensorflow as tf #for building and training deep learning models
import matplotlib.pyplot as plt #for visualising images and plots
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import normalize

values = [7,2,9,8,5,1]
correct_predictions = 0
incorrect_predictions = 0

#Trains a new model or load an existing one
train_new_model = True

if train_new_model:

    #Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Normalize and reshape the data for CNN input - CNN expects inputs of (batch_size, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    #Scaling pixel values to range [0,1] for better training efficiency

    #Define the CNN model

    model = Sequential([
        Input(shape=(28, 28, 1)),  #Input layer (28x28 grayscale image)

        # Feature extraction layers
        Conv2D(32, (3, 3), activation='relu'),  #Detects edges and patterns
        MaxPooling2D(pool_size=(2, 2)),  #Reduces spatial size (downsampling) - reduces size of image/feature map
        Dropout(0.25),  #Prevents overfitting (learning data too well but performing poorly on new data)

        Conv2D(64, (3, 3), activation='relu'),  #Detects more complex shapes
        MaxPooling2D(pool_size=(2, 2)),  #Further downsampling
        Dropout(0.25),  #Regularization to reduce overfitting

        Flatten(),  #Converts 2D feature maps to 1D vector

        #Classification layers
        Dense(128, activation='relu'),  #Fully connected layer
        Dropout(0.5),  #Regularization
        Dense(10, activation='softmax')  #Output layer (10 classes for digits 0-9)
    ])

    #Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Train the model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    #Saving the trained model
    model.save('cnn_handwritten_digits.keras')

    #Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
else:
    model = tf.keras.models.load_model('cnn_handwritten_digits.keras')


#Load custom images and predict
def predict_custom_images(correct_predictions, incorrect_predictions):
    image_number = 1
    while os.path.isfile(f'digits/digit{image_number}.png'):
        try:
            #Load and preprocess custom images
            img = cv2.imread(f'digits/digit{image_number}.png', cv2.IMREAD_GRAYSCALE) #Reads image in greyscale to match MNIST regulations
            img = cv2.resize(img, (28, 28)) #Resizes the image to 28x28 pixels
            img = np.invert(img) #Inverts the image - white digits and black background
            img_array = np.array(img) / 255.0  #Normalize - scales pixel values to [0,1]
            img_array = img_array.reshape(1, 28, 28, 1)  #Reshape for CNN input

            #Predict using the trained CNN model
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            print(f"Image digit{image_number}.png: Predicted Digit = {predicted_digit}")
            if predicted_digit == values[image_number-1]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

            #Display the image
            plt.imshow(img_array[0, :, :, 0], cmap=plt.cm.binary)
            plt.title(f"Prediction: {predicted_digit}")
            plt.show()
            input("Press Enter to proceed to the next image...")


            image_number += 1
        except Exception as e:
            print(f"Error reading image digit{image_number}.png: {e}")
            image_number += 1

    return correct_predictions, incorrect_predictions


#Predict custom images
correct_predictions, incorrect_predictions = predict_custom_images(correct_predictions, incorrect_predictions)
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {incorrect_predictions}")
print("Accuracy: " ,(correct_predictions / 6 )*100, "%")