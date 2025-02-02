import os
import cv2
import numpy as np
from datetime import datetime
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


def select_images():
    """Open file dialog to select two images for comparison."""
    root = tk.Tk()
    root.withdraw()
    root.update()  # Ensure proper initialization
    file_paths = filedialog.askopenfilenames(
        title="Select two images",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    root.destroy()  # Ensure Tkinter closes properly

    if len(file_paths) != 2:
        messagebox.showerror("Error", "Please select exactly two images.")
        return None, None

    return file_paths[0], file_paths[1]


def safe_normalize(vec, eps=1e-10):
    """Safely normalize a vector to avoid division by zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > eps else vec


def extract_lbp_features(image, radius=3, n_points=24, method='uniform'):
    """Extract Local Binary Pattern (LBP) features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    # Use a basic histogram (no double normalization), then safely normalize
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),  # n_points+2 bins, +1 for range
        density=False
    )
    return safe_normalize(hist)


def extract_glcm_features(image, distances=None, angles=None):
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) features from an image.
    Using default distance=[1], angle=[0].
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    # For simplicity, just take the first distance/angle combination
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    features = np.array([
        contrast, dissimilarity, homogeneity, energy, correlation
    ])
    return safe_normalize(features)


def extract_gabor_features(image, frequency=0.6):
    """Extract Gabor filter features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_real, gabor_imag = gabor(gray, frequency=frequency)
    # Create a feature vector and normalize
    features = np.array([
        gabor_real.mean(),
        gabor_real.var(),
        gabor_imag.mean(),
        gabor_imag.var()
    ])
    return safe_normalize(features)


def extract_entropy(image):
    """Extract entropy feature from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a simple histogram (no double normalization), then compute entropy
    hist, _ = np.histogram(gray.ravel(), bins=256, density=False)
    entr = entropy(hist)
    return safe_normalize(np.array([entr]))


def resize_images_to_same_height_with_gap(img1, img2, gap=10):
    """Resize images to the same height and add a gap between them."""
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    target_height = min(height1, height2)

    img1_resized = cv2.resize(
        img1,
        (int(width1 * (target_height / height1)), target_height)
    )
    img2_resized = cv2.resize(
        img2,
        (int(width2 * (target_height / height2)), target_height)
    )

    gap_array = np.ones((target_height, gap, 3), dtype=np.uint8) * 255  # White gap
    combined_image = np.hstack((img1_resized, gap_array, img2_resized))

    return combined_image


def compare_textures(image1, image2, image_path1, image_path2):
    """Compare two images based on texture descriptors and return a similarity."""
    lbp1, lbp2 = extract_lbp_features(image1), extract_lbp_features(image2)
    glcm1, glcm2 = extract_glcm_features(image1), extract_glcm_features(image2)
    gabor1, gabor2 = extract_gabor_features(image1), extract_gabor_features(image2)
    entropy1, entropy2 = extract_entropy(image1), extract_entropy(image2)

    # Compute pairwise distances
    lbp_distance = euclidean(lbp1, lbp2)
    glcm_distance = euclidean(glcm1, glcm2)
    gabor_distance = euclidean(gabor1, gabor2)
    entropy_distance = euclidean(entropy1, entropy2)

    print("\n--- Texture Analysis ---")
    print(f"LBP Distance: {lbp_distance:.4f}")
    print(f"GLCM Distance: {glcm_distance:.4f}")
    print(f"Gabor Distance: {gabor_distance:.4f}")
    print(f"Entropy Distance: {entropy_distance:.4f}")

    # Convert total distance into a similarity measure via exponential decay
    total_distance = lbp_distance + glcm_distance + gabor_distance + entropy_distance
    similarity_score = np.exp(-total_distance) * 100  # scale to percentage

    print(f"Computed Similarity Score: {similarity_score:.2f}%\n")

    # --- LOG THE RESULTS TO A FILE WITH DATE/TIME AND FILENAMES ---
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    image_name1 = os.path.basename(image_path1)
    image_name2 = os.path.basename(image_path2)
    with open("texture_comparison_log.txt", "a", encoding='utf-8') as log_file:
        log_file.write(f"--- Log Entry: {now} ---\n")
        log_file.write(f"Image 1: {image_name1}\n")
        log_file.write(f"Image 2: {image_name2}\n")
        log_file.write(f"LBP Distance: {lbp_distance:.4f}\n")
        log_file.write(f"GLCM Distance: {glcm_distance:.4f}\n")
        log_file.write(f"Gabor Distance: {gabor_distance:.4f}\n")
        log_file.write(f"Entropy Distance: {entropy_distance:.4f}\n")
        log_file.write(f"Computed Similarity Score: {similarity_score:.2f}%\n\n")

    return similarity_score


def load_and_compare():
    """Load two images, compare them, and display the results."""
    image_path1, image_path2 = select_images()
    if not image_path1 or not image_path2:
        return

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        messagebox.showerror("Error", "One or both images could not be loaded.")
        return

    similarity = compare_textures(image1, image2, image_path1, image_path2)
    print(f"Texture Similarity Score: {similarity:.2f}%")
    messagebox.showinfo("Similarity Score", f"Texture Similarity Score: {similarity:.2f}%")

    # Show combined image
    combined_image = resize_images_to_same_height_with_gap(image1, image2)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Texture Similarity: {similarity:.2f}%")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(10)  # Keep figure open for 10 seconds
    plt.close()


if __name__ == "__main__":
    load_and_compare()  