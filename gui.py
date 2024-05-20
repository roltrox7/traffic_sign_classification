import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('/Users/sampreethshetty/Documents/chotu/its/my_model.h5')

# Dictionary to label all traffic signs class
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing vehicle with a weight greater than 3.5 tons'}

# Constants
IMG_HEIGHT, IMG_WIDTH = 30, 30  # Change these based on your model's input shape

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load and classify image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0] + 1  # Class index starts from 1 in the classes dict
        result_label.config(text=f'Predicted Class: {classes[predicted_class]}')
        
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize for display in the GUI
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

# Set up the GUI
root = tk.Tk()
root.title("Traffic Sign Classifier")
root.geometry("500x500")  # Set the window size

# Add a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=20)

# Label to display the uploaded image
img_label = Label(root)
img_label.pack(pady=20)

# Label to display the prediction result
result_label = Label(root, text="Predicted Class: None", font=('Helvetica', 14))
result_label.pack(pady=20)

# Run the application
root.mainloop()
