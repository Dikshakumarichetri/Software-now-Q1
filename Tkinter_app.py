import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
import os

# Loading the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")


class GUIElements:
    def __init__(self):
        self.img_lbl = None  # Label for the image display
        self.result_label = None  # Label for the classification result
        self.photo_img = None  # To hold the reference to the uploaded image

    def create_header(self, root):
        header_frame = tk.Frame(root, bg="#1f2a38")
        header_frame.pack(fill="x", pady=20)

        header_label = tk.Label(
            header_frame,
            text="Image Classifier Studio",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#1f2a38",
            pady=10,
        )
        header_label.pack()

    def create_footer(self, root):
        footer_frame = tk.Frame(root, bg="#1f2a38")
        footer_frame.pack(fill="x", side="bottom", pady=10)

        footer_label = tk.Label(
            footer_frame,
            text="Powered by MobileNetV2",
            font=("Arial", 10, "italic"),
            fg="white",
            bg="#1f2a38",
            pady=5,
        )
        footer_label.pack()

    def create_input_area(self, root):
        input_frame = tk.Frame(root, bg="#2c3e50")
        input_frame.pack(pady=20)

        input_label = tk.Label(
            input_frame,
            text="Upload an image for classification:",
            font=("Arial", 14),
            bg="#2c3e50",
            fg="white",
        )
        input_label.pack(pady=5)

    def create_buttons(self, root, upload_action, classify_action, clear_action):
        button_frame = tk.Frame(root, bg="#2c3e50")
        button_frame.pack(pady=20)

        # Improved button style with hover effect
        def on_enter(e, button):
            button["background"] = "#2980b9"

        def on_leave(e, button):
            button["background"] = "#3498db"

        # Upload button
        upload_button = tk.Button(
            button_frame,
            text="Upload Image",
            command=upload_action,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="grey",
            padx=30,
            pady=12,
            borderwidth=0,
            relief="flat",
        )
        upload_button.grid(row=0, column=0, padx=20)
        upload_button.bind("<Enter>", lambda e: on_enter(e, upload_button))
        upload_button.bind("<Leave>", lambda e: on_leave(e, upload_button))

        # Improved button style with hover effect for classify
        def on_enter_classify(e, button):
            button["background"] = "#218c3c"

        def on_leave_classify(e, button):
            button["background"] = "#28a745"

        # Classify button
        classify_button = tk.Button(
            button_frame,
            text="Classify Image",
            command=classify_action,
            font=("Arial", 12, "bold"),
            bg="#28a745",
            fg="grey",
            padx=30,
            pady=12,
            borderwidth=0,
            relief="flat",
        )
        classify_button.grid(row=0, column=1, padx=20)
        classify_button.bind("<Enter>", lambda e: on_enter_classify(e, classify_button))
        classify_button.bind("<Leave>", lambda e: on_leave_classify(e, classify_button))

        # Clear button for multiple image classification
        def on_enter_clear(e, button):
            button["background"] = "#c0392b"

        def on_leave_clear(e, button):
            button["background"] = "#e74c3c"

        clear_button = tk.Button(
            button_frame,
            text="Clear Image",
            command=clear_action,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="grey",
            padx=30,
            pady=12,
            borderwidth=0,
            relief="flat",
        )
        clear_button.grid(row=0, column=2, padx=20)
        clear_button.bind("<Enter>", lambda e: on_enter_clear(e, clear_button))
        clear_button.bind("<Leave>", lambda e: on_leave_clear(e, clear_button))

    def display_image(self, root, image_path):
        if self.img_lbl:
            self.img_lbl.pack_forget()  # Remove previous image if any

        # Open the image and resize it for preview
        img = Image.open(image_path)
        img = img.resize(
            (250, 250), Image.Resampling.LANCZOS
        )  # Use LANCZOS for smooth resizing
        self.photo_img = ImageTk.PhotoImage(
            img
        )  # Store image reference to prevent garbage collection

        # Create or update image label
        self.img_lbl = tk.Label(root, image=self.photo_img, bg="#2c3e50")
        self.img_lbl.image = self.photo_img
        self.img_lbl.pack(pady=10)

        root.update()

    def display_result(self, root, result):
        if self.result_label:
            self.result_label.pack_forget()

        self.result_label = tk.Label(
            root, text=result, font=("Arial", 14), fg="#dc3545", bg="#2c3e50"
        )
        self.result_label.pack(pady=10)


# Real image classifier function using MobileNetV2
def real_image_classifier(image_path):
    try:
        img = Image.open(image_path).resize((224, 224))  # MobileNetV2 input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)  # Get model predictions
        decoded_predictions = decode_predictions(predictions, top=1)[0][
            0
        ]  # Decode top prediction

        label = decoded_predictions[1]  # Human-readable label
        confidence = decoded_predictions[2]  # Confidence score

        return f"Classified as: {label} with {confidence:.2f} confidence."
    except Exception as e:
        return f"Error: {str(e)}"


# Base class for the application
class ApplicationBase:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier Studio")
        self.root.geometry("700x600")  # Set window size
        self.root.configure(bg="#2c3e50")  # Dark background

    def start(self):
        self.root.mainloop()


# GUI class responsible for creating the buttons and displaying images


# Main application class
class ImageClassifierApp(ApplicationBase, GUIElements):
    def __init__(self, root):
        ApplicationBase.__init__(self, root)  # Calling constructor of ApplicationBase
        GUIElements.__init__(self)  # Calling constructor of GUIElements

        self.selected_image_path = None  # Encapsulation: Private variable

        self.create_header(root)  # Create a header
        self.create_input_area(root)  # Instructional area
        self.create_buttons(
            root, self.upload_image, self.classify_image, self.clear_image
        )
        self.create_footer(root)  # Create a footer

    # Method Overriding: Override the behavior for uploading an image
    def upload_image(self):
        file_path = filedialog.askopenfilename()

        if file_path and os.path.exists(file_path):
            try:
                Image.open(file_path).verify()  # Verifying if it's a valid image
                self.selected_image_path = file_path
                self.display_image(
                    self.root, file_path
                )  # Displaying the uploaded image preview
            except (IOError, SyntaxError) as e:
                messagebox.showerror("Error", f"Invalid image file selected: {str(e)}")
        else:
            messagebox.showerror("Error", "No valid image file selected.")

    # Polymorphism: Same method for any image, but handles different inputs
    def classify_image(self):
        if not self.selected_image_path:
            messagebox.showerror("Error", "No image selected!")
        else:
            result = real_image_classifier(self.selected_image_path)
            self.display_result(self.root, result)

    # Clear image for multiple classifications
    def clear_image(self):
        self.selected_image_path = None
        if self.img_lbl:
            self.img_lbl.pack_forget()  # Remove image
            self.img_lbl = None  # Reset label
        if self.result_label:
            self.result_label.pack_forget()  # Remove result
            self.result_label = None  # Reset label


# Error handling and logging decorators
def logger_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling method: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def error_handler_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error occurred: {e}")
            messagebox.showerror("Error", str(e))

    return wrapper


# Applying both decorators to the main function
@logger_decorator
@error_handler_decorator
def main():
    root = tk.Tk()
    app = ImageClassifierApp(root)  # Creating the Image Classifier App
    app.start()  # Starting the Tkinter main loop


if __name__ == "__main__":
    main()
