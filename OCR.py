import cv2
import pytesseract
import os

# Path to the folder containing reCAPTCHA images
image_folder = "/home/arslan/DIP Projects/FINAL PROJECT/MobileNet Fine-tuning/train/22"

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Remove noise (optional, depending on image quality)
    # Use morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return clean_image

def extract_text_from_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(processed_image)
    return text

# Process and extract text from all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        text = extract_text_from_image(image_path)
        print(f'Extracted text from {filename}: {text}')
