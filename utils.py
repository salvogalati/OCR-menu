from io import BytesIO
from PIL import Image
import tempfile, traceback, base64
import dspy
from tqdm import tqdm
import requests
from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

# Pixel limits for acceptable image sizes
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

# Encode image to base64 from path

def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Resize and encode image, keeping resolution within bounds

def resize_encode_image(image_path, min_pixels, max_pixels):
    if not isinstance(image_path, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(image_path)
            image_path = temp_pdf.name

    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    if total_pixels <= max_pixels:
        print(f"The image {image_path} is already within the range ({width}x{height}).")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        scale_factor = (max_pixels / total_pixels) ** 0.5

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    img_bytes = BytesIO()
    resized_img.save(img_bytes, format="PNG")
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

# Convert each PDF page to base64-encoded PNG

def encode_pdf(path: str, page_indexes: tuple = None):
    if not isinstance(path, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(path)
            path = temp_pdf.name

    reader = PdfReader(path)
    pages_base64 = []
    num_pages = len(reader.pages)

    if page_indexes is None:
        page_indexes = range(num_pages)

    for i in tqdm(page_indexes):
        pages_base64.append(render_pdf_to_base64png(path, i, target_longest_image_dim=1024))

    return pages_base64

# Process file upload and send OCR request to backend

def process_file(data_input, file_extension, API_URL):
    data = []
    output = []
    try:
        # Encode to base64 depending on file type
        if file_extension == ".pdf":
            data.extend(encode_pdf(data_input))
        else:
            data.append(resize_encode_image(data_input, max_pixels=max_pixels, min_pixels=min_pixels))

        # Send request to backend API for each image/page
        for page in data:
            payload = {"data": page}
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                import streamlit as st
                st.error("Error during image processing. Please try again.")
                st.write(response.json())
                continue
            else:
                output.append(response.json()["data"])

    except Exception as e:
        output = f"Error processing image: {traceback.format_exc()}"

    finally:
        return output

# DSPy signature for menu parsing from raw OCR text
class BasicQA(dspy.Signature):
    """Format this menu into a list readble for pandas dataframe.
    The dataframe should have three columns: name, ingredients, price, section.
    If there is no section, you must assign one from: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande.
    Section MUST be one of these categories. Do not split dishes on the same row.
    """
    question = dspy.InputField(desc="Raw menu input")
    answer: list = dspy.OutputField(desc="should be a list")
