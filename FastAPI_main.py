import os
from fastapi import FastAPI, Request, HTTPException
import socket
import traceback

import torch
import base64

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import gc
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

# Ensure memory allocation for large tensors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Helper to encode image to base64

def encode_image_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Resize and encode image to base64 while preserving aspect ratio

def resize_encode_image(image_path, min_pixels, max_pixels):
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    if min_pixels <= total_pixels <= max_pixels:
        print(f"The image {image_path} is already within the range ({width}x{height}).")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    if total_pixels > max_pixels:
        scale_factor = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        img_bytes = BytesIO()
        resized_img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    raise ValueError("Image too small, below minimum pixel threshold.")

# Convert PDF pages to base64-encoded images

def encode_pdf(path: str, page_indexes: tuple = None):
    if not isinstance(path, str):  # FIXED: was incorrectly checking `image_path`
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

# OCR processing of an input base64 image

def OCR_menu(image_data):
    try:
        del output
        del inputs
    except:
        pass
    torch.cuda.empty_cache()
    gc.collect()

    image_base64 = image_data

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """Extract only the menu dishes from the image and provide it as a list
                with following fields: name, price, ingredients.
                Try to dectect the section of each dish among the following categories: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande.
                If any of these fields are not present use the value None.
                DO NOT PROVIDE ANY OTHER INFORMATION
                Example of output: [
                    ['Pizza Margherita', '10.0', 'Pomodoro, Mozzarella', "Secondi"],
                    ["Pasta al pomodoro", '8.00', 'Pasta, pomodoro, basilico', "Primi"]
                ]"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=8000,
        num_return_sequences=1,
        do_sample=True,
    )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return text_output

# Load OCR model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval()

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI setup
app = FastAPI(
    title="OCR API",
    description="This API processes images and extracts text using OCR.",
    version="1.0.0"
)

# Test route
@app.get("/")
def index():
    hostname = socket.gethostname()
    return {"message": f"Hello from Colab! I am the hostname {hostname}"}

# OCR route
@app.post("/api/", summary="Perform OCR", description="Processes an image and returns raw extracted text.")
async def ocr_image(request: Request):
    print("Processing request")
    try:
        data = await request.json()
        res = OCR_menu(data['data'])
        return {
            "status": "success",
            "message": "Data retrieved successfully",
            "data": res[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": traceback.format_exc(),
            "data": None
        })