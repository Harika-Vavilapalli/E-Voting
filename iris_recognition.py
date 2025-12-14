import os
import cv2
import numpy as np

# Path to dataset
DATASET_PATH = "C:/Users/Harika/Desktop/e_voting/iris_dataset/iris_images"

def match_iris(image_path):
    """
    Compare the uploaded image with images in the dataset.
    Returns "Match Found!" if a match is found, otherwise "Match Not Found!".
    """
    # Load the uploaded image in grayscale
    input_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        return "Invalid Image"

    # Resize the input image to a fixed size for consistent comparison
    input_img = cv2.resize(input_img, (64, 64))

    # Initialize variables to track the best match
    best_similarity = -1
    best_match = "Match Not Found!"

    # Iterate through the dataset
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        for eye_side in ["left", "right"]:
            side_path = os.path.join(folder_path, eye_side)
            if not os.path.exists(side_path):
                continue

            # Iterate through images in the side folder
            for img_name in os.listdir(side_path):
                img_path = os.path.join(side_path, img_name)
                dataset_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if dataset_img is None:
                    continue

                # Resize the dataset image to match the input image size
                dataset_img = cv2.resize(dataset_img, (64, 64))

                # Compare histograms for similarity
                hist1 = cv2.calcHist([input_img], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([dataset_img], [0], None, [256], [0, 256])
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

                # Update the best match if the current similarity is higher
                if similarity > best_similarity:
                    best_similarity = similarity

                # If similarity exceeds the threshold, return "Match Found!"
                if similarity > 0.9:  # Threshold for match
                    return "Match Found!"

    # If no match is found, return the best similarity score for debugging
    return f"Match Not Found! (Best Similarity: {best_similarity:.2f})"