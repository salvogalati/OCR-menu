import getpass
import os
import threading
from fastapi import FastAPI, Request, HTTPException
#from pyngrok import ngrok, conf
import socket
import traceback
#import nest_asyncio
#import uvicorn
from pydantic import BaseModel

import torch
import base64
import urllib.request

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import json, re, ast
import pandas as pd
import dspy
import gc
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from pypdf import PdfReader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_encode_image(image_path, min_pixels, max_pixels):
    # Open the image
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate the total number of pixels
    total_pixels = width * height

    # If the image is already within the range, save it as is
    if min_pixels <= total_pixels <= max_pixels:
        print(f"The image {image_path} is already within the range ({width}x{height}).")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Compute the scaling factor
    if total_pixels > max_pixels:
        scale_factor = (max_pixels / total_pixels) ** 0.5  # Downscale the image

    # Compute the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image while maintaining the aspect ratio
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    resized_img.save("test.png")
    img_bytes = BytesIO()
    resized_img.save(img_bytes, format="PNG")  # Save as JPEG (change if needed)
    img_bytes = img_bytes.getvalue()
    
    # Encode to Base64
    return base64.b64encode(img_bytes).decode('utf-8')

def encode_pdf(path: str, page_indexes: tuple = None):

    if not isinstance(image_path):
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


def OCR_menu(image_data):
    #Clean the memory
    try:
        del output
        del inputs
    except:
        pass
    torch.cuda.empty_cache()

    image_base64 = image_data

    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Extract only the menu dishes from the image and provide it as a list\
                         with following fields: name, price, ingredients.
                         Try to dectect the section of each dish among the following categories: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande
                        If any of these fields are not present use the value None
                        DO NOT PROVIDE ANY OTHER INFORMATION
                        Example of output: [
                            ['Pizza Margherita', '10.0', 'Pomodoro, Mozzarella', "Secondi"]
                         ["Pasta al pomodoro", '8.00', 'Pasta, pomodoro, basilico', "Primi"]
                         ]"""},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"} },
                    ],
                }
            ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
    inputs = {key: value.to(device) for (key, value) in inputs.items()}


    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=8000,
                num_return_sequences=1,
                do_sample=True,
        )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )

    return text_output

model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Create FastAPI instance with metadata
app = FastAPI(
    title="OCR API",
    description="This API processes images and extracts text using OCR.",
    version="1.0.0"
)

"""
# Configure ngrok authtoken
conf.get_default().auth_token = "2tfICKs9np8t7zJv5SaZwZBzz8F_6UiDQvyq3UoB58haqpuKN"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(8000).public_url  # FastAPI uses port 8000 by default
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:8000/\"")

# Update the base URL with the public ngrok URL
app.state.base_url = public_url
"""

# Define FastAPI routes
@app.get("/")
def index():
    hostname = socket.gethostname()
    return {"message": f"Hello from Colab! I am the hostname {hostname}"}

@app.post("/api/", summary="Perform OCR", description="Processes an image and returns raw extracted text.")
async def ocr_image(request: Request):
    print("Processing request")
    try:
        # Extract JSON data from the request
        data = await request.json()
        res = OCR_menu(data['data'])  # OCR_menu function must be defined elsewhere
        menu_data = {
            "status": "success",
            "message": "Data retrieved successfully",
            "data": res[0]
        }
        return menu_data  # FastAPI automatically converts the dictionary to JSON
    except Exception as e:
        error_data = {
            "status": "error",
            "message": traceback.format_exc(),
            "data": None
        }
        raise HTTPException(status_code=400, detail=error_data)



#nest_asyncio.apply()
#uvicorn.run(app, port=8000)