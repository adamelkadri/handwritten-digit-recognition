import os #handles file operations - useful to test images
import cv2 #OpenCV library for image processing
import numpy as np #for numerical operations
import tensorflow as tf #for building and training deep learning models
import matplotlib.pyplot as plt #for visualising images and plots




values = [7,2,9,8,5,1]
correct_predictions = 0
incorrect_predictions = 0

#Decide if to load an existing model or to train a new one
train_new_model = True


if train_new_model:
    #Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist

    #Loading the data and creating two tuples - x is for pixels and y is for the classification (numbers between 0-9)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Normalizing the x data only (making length = 1) - improves training stability
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    #Create a sequential neural network model - linear stack of layers
    model = tf.keras.models.Sequential()

    #Flatten layer reshapes data into 1D vector - converts 28x28 into a singular vector of 784 values
    model.add(tf.keras.layers.Flatten())

    #Dense Layer containing 128 neurons, and ReLU activation ensures efficient training to eliminate vanishing gradients
    #Stacking two dense layers to increase network's capacity to learn more complex features
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

    #Dense layer with 10 units representing the numbers from 0-9, softmax activation converts output values into probabilities
    #Each unit's probability sums to 1 - highest probability determines the predicted class
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #Uses an optimization algorithm to adjust the learning rate dynamically for each parameter
    # - calculates first moment (mean of gradients) and second moment (uncentered variance)
    #Uses a loss function to measure how well the model's predictions are against the actual labels


    # Training the model
    model.fit(X_train, y_train, epochs=5) #5 complete passes through the entire training dataset

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.keras')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Load custom images and predict them
def predict_custom_images(correct_predictions, incorrect_predictions):
    images = []
    titles = []
    image_number = 1
    while os.path.isfile('digits/digit{}.png'.format(image_number)):
        try:
            img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction)
            print(f"Image digit{image_number}.png: Predicted Digit = {predicted_digit}")

            if predicted_digit == values[image_number - 1]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

            images.append(img[0])
            titles.append(f"Prediction: {predicted_digit}")
            image_number += 1
        except:
            print("Error reading image! Proceeding with next image...")
            image_number += 1

    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 3))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=plt.cm.binary)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return correct_predictions, incorrect_predictions

correct_predictions, incorrect_predictions = predict_custom_images(correct_predictions, incorrect_predictions)
print(f"Correct predictions: {correct_predictions} ")
print(f"Incorrect predictions: {incorrect_predictions} ")
print("Accuracy:", (correct_predictions / 6) * 100, "%")