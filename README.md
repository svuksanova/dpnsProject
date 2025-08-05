# Texture Similarity App

A Python-based desktop application that compares two images and computes their texture similarity using classical texture descriptors: **LBP**, **GLCM**, **Gabor filters**, and **Entropy**.  
The final similarity score is calculated using Euclidean distance and displayed in a simple, user-friendly interface.


## 🧠 Features

- 📷 Select and compare any two images
- 🔍 Uses four powerful descriptors:
  - **LBP** (Local Binary Pattern)
  - **GLCM** (Gray-Level Co-occurrence Matrix)
  - **Gabor filters**
  - **Entropy**
- 📊 Computes a unified similarity score (%)
- 💾 Automatically logs comparisons in a `.txt` file
- 🖼 Visual side-by-side comparison with Matplotlib
- 🪟 Simple GUI built with `tkinter`


## 📄 Documentation

For a full explanation of the algorithms, implementation details, and theoretical background, see the full project report:

📘 [Documentation (PDF)](./Dokumentacija.pdf)


## 🛠 Technologies Used

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Scikit-Image
- SciPy
- Matplotlib
- Tkinter (for GUI)


## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/svuksanova/dpnsProject.git
cd dpnsProject
```

2. Install dependencies:
```bash
pip install opencv-python numpy scikit-image matplotlib scipy
```

3. Run the main script:
```bash
python main.py
```

