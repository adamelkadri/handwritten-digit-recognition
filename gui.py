import tkinter as tk #gui toolkit for python
from tkinter import Canvas
import numpy as np #for numerical operations
from PIL import Image, ImageOps, ImageDraw #for image pre-processing
import tensorflow as tf #used to load pre-trained cnn model

#Load the trained CNN model
model = tf.keras.models.load_model('cnn_handwritten_digits.keras')

#Create main application window
app = tk.Tk()


#Opens window above all other windows
app.lift()
app.attributes('-topmost', True)
app.after(500, lambda: app.attributes('-topmost', False))


#Displays window in the centre of screen
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

window_width = 400
window_height = 400
x_position = (screen_width // 2) - (window_width // 2)
y_position = (screen_height // 2) - (window_height // 2)

app.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
app.title("Digit Recognizer")


#Resizing window if necessary
app.rowconfigure(0, weight=1)
app.rowconfigure(1, weight=1)
app.rowconfigure(2, weight=1)
app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=1)

# Create canvas for drawing
canvas = Canvas(app, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")  #center canvas properly


image = Image.new("L", (280, 280), "white") #creates blank grayscale image to store drawn digit
draw_area = ImageDraw.Draw(image) #initialises draw area to draw input



#Handles drawing on the canvas
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-8, y-8, x+8, y+8, fill="black", width=5)
    draw_area.ellipse([x-8, y-8, x+8, y+8], fill="black") #updates in-memory image with drawn stroke for processing


#Clears canvas and resets in-memory image for a new drawing
def clear_canvas():
    canvas.delete("all")
    global image, draw_area
    image = Image.new("L", (280, 280), "white")
    draw_area = ImageDraw.Draw(image)


#Processes drawn digit
def preprocess_image():
    img = image.convert('L') #converts image to greyscale - 'L' means luminance so shades of grey in range 0-255
    img = ImageOps.invert(img)  #invert colors (black on white)
    bbox = img.getbbox()  #get bounding box of the digit - smallest rectangle that contains all non-white pixels
    if bbox:
        img = img.crop(bbox)
        img = ImageOps.pad(img, (28, 28), method=Image.LANCZOS, color = "white")
    img = img.resize((28, 28), Image.LANCZOS)  #resize to 28x28
    img_array = np.array(img) / 255.0  #normalize pixel values to the range [0,1] for model input
    return img_array.reshape(1, 28, 28, 1)


#Uses trained model to predict digit drawn
def predict():
    img_array = preprocess_image()
    prediction = model.predict(img_array) #uses tensorflow model to predict digit drawn on canvas
    digit = np.argmax(prediction)
    confidence = prediction[0][digit] * 100
    label_result.config(text=f"Digit: {digit} ({confidence:.2f}%)")

canvas.bind("<B1-Motion>", draw)

# Buttons for clearing and predicting
btn_clear = tk.Button(app, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=0, pady=10)

btn_predict = tk.Button(app, text="Predict", command=predict)
btn_predict.grid(row=1, column=1, pady=10)

# Label to display result
label_result = tk.Label(app, text="Draw a digit and predict", font=("Arial", 14))
label_result.grid(row=2, column=0, columnspan=2, pady=10)

#opens app
app.update_idletasks()
app.deiconify()

app.mainloop()
