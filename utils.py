from io import BytesIO
from PIL import Image
import json, re, ast, tempfile, os, traceback, base64
import pandas as pd
import dspy
from tqdm import tqdm
import requests
from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

min_pixels = 256*28*28
max_pixels = 1280*28*28

def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_encode_image(image_path, min_pixels, max_pixels):
    
    if not isinstance(image_path, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(image_path)
            image_path = temp_pdf.name
            
    # Open the image
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate the total number of pixels
    total_pixels = width * height

    # If the image is already within the range, save it as is
    if total_pixels <= max_pixels:
        print(f"The image {image_path} is already within the range ({width}x{height}).")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else: # Compute the scaling factor
        scale_factor = (max_pixels / total_pixels) ** 0.5  # Downscale the image
    

    # Compute the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image while maintaining the aspect ratio
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    #resized_img.save("test.png")
    img_bytes = BytesIO()
    resized_img.save(img_bytes, format="PNG")  # Save as JPEG (change if needed)
    img_bytes = img_bytes.getvalue()
    
    # Encode to Base64
    return base64.b64encode(img_bytes).decode('utf-8')

def encode_pdf(path: str, page_indexes: tuple = None):

    if not isinstance(path, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(path)
            path = temp_pdf.name

    reader = PdfReader(path)
    pages_base64 = []
    num_pages = len(reader.pages)
    # If pages is None, process all pages
    if page_indexes is None:
        page_indexes = range(num_pages)  # Default: all pages
    
    for i in tqdm(page_indexes):
        pages_base64.append(render_pdf_to_base64png(path, i, target_longest_image_dim=1024))

    return pages_base64

# Function to process the uploaded image
def process_file(data_input, file_extension, API_URL):
    data = []
    output = []
    try:
        # Encode as Base64
        if file_extension == ".pdf":
            data.extend(encode_pdf(data_input))
        else:
            data.append(resize_encode_image(data_input, max_pixels=max_pixels, min_pixels=min_pixels))

        for page in data:
            payload = {"data": page}  # Create the JSON payload containing the Base64-encoded image data
            # Send a POST request to the API with the JSON payload
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                st.error("Error during image processing. Please try again.")
                st.write(response.json())
                continue
            else:
                output.append(response.json()["data"])
    except Exception as e:
        output = f"Error processing image: {traceback.format_exc()}"
    finally: 
        return output


#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Format this menu into a list readble for pandas dataframe.
    The datafram should have tre columns: name, ingridients, price, section
    If there is no section, you have to decide the section of each dish among the following categories: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande
    The section MUST BE ONE OF THESE: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande
    Do not split the dish if it is in the same row
    """
    question = dspy.InputField(desc="Raw menu input")
    answer: list = dspy.OutputField(desc="should be a list")