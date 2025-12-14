from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from iris_recognition import match_iris  # Import the matching function

app = Flask(__name__, template_folder="templates")

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload and iris matching."""
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")

    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Check if the uploaded image matches any in the dataset
        result = match_iris(file_path)
        return render_template("result.html", result=result)
    else:
        return render_template("index.html", error="Invalid file type. Please upload a PNG, JPG, or JPEG file.")

if __name__ == "__main__":
    app.run(debug=True)