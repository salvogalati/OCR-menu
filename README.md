# OCR Menu Extraction Project

This project leverages **FastAPI**, **Streamlit**, and machine learning models to extract and optimize restaurant menus from images or PDFs. The goal is to identify dishes, prices, ingredients, and sections in a structured format.

## Project Structure

- **`FastAPI_main.py`**: Implements a REST API for processing images and PDFs to extract text.
- **`Streamlit_main.py`**: Provides a user interface for uploading files and visualizing the extracted data.
- **`utils.py`**: Contains utility functions for encoding images and PDFs into Base64 and sending requests to the API.
- **`olmOCR.ipynb`**: A Jupyter Notebook for testing and developing the OCR extraction model.
- **`Menu-OCR_detection/`**: Directory for additional resources or models.

## Features

- Extracts menu data (dish name, price, ingredients, and section) from images or PDFs.
- Supports optimization of extracted data using language models like **LLama 3 8B** or **DeepSeek API**.
- Provides a web-based interface for easy interaction and visualization.

## Requirements

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- Required Python libraries:
  - `fastapi`
  - `streamlit`
  - `torch`
  - `transformers`
  - `pillow`
  - `pypdf[full]`
  - `dspy`
  - `tqdm`
  - `requests`

## Installation

1. Clone the repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_FOLDER>
   ```

2. Install the required dependencies:

   If using pip:
   ```bash
   pip install -r requirements.txt
   ```

   If using conda:
   ```bash
   conda env create -f environment.yml
   conda activate ocr-menu-app
   ```

3. Install `olmocr` following the official repository instructions:
   > 📦 https://github.com/allenai/olmocr

   Example:
   ```bash
   git clone https://github.com/allenai/olmocr.git
   cd olmocr
   pip install -e .
   ```

## Usage

You can use this project in **three ways**:

---

### 1. 🧪 Notebook (Exploration & Testing)

Use the `olmOCR.ipynb` notebook to test and prototype OCR extraction manually.

```bash
jupyter notebook olmOCR.ipynb
```

---

### 2. 🚀 FastAPI Server (Backend API)

Launch the FastAPI server to expose a REST API for OCR processing:

```bash
uvicorn FastAPI_main:app --reload --host 0.0.0.0 --port 8000
```

This will start the API at `http://localhost:8000/api/`

You can POST base64-encoded image data to this endpoint using tools like `curl` or Python `requests`.

---

### 3. 🌐 Streamlit Web App (User Interface)

Start the front-end UI to upload menus and visualize structured results:

```bash
streamlit run Streamlit_main.py
```

You’ll be able to choose between **LLama 3 8B** and **DeepSeek** for post-processing the OCR results.

> ⚠️ Make sure the FastAPI server is running before launching the web app.

---

### 🔐 API Tokens Required

Before running the app, you **must set your API tokens manually** in the code.

Open the file `Streamlit_main.py` and set the following variables:

```python
huggingface_token = "your_huggingface_token_here"
deep_seek_token = "your_deepseek_token_here"
```

These tokens are used to call external language models like LLama 3 or DeepSeek API.

> 💡 In future versions, these may be loaded from a `.env` file.

---