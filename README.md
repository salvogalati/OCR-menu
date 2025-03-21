# OCR Menu Extraction Project

This project leverages **FastAPI**, **Streamlit**, and machine learning models to extract and optimize restaurant menus from images or PDFs. The goal is to identify dishes, prices, ingredients, and sections in a structured format.

## Project Structure

- **[`FastAPI_main.py`](FastAPI_main.py)**: Implements a REST API for processing images and PDFs to extract text.
- **[`Streamlit_main.py`](Streamlit_main.py)**: Provides a user interface for uploading files and visualizing the extracted data.
- **[`utils.py`](utils.py)**: Contains utility functions for encoding images and PDFs into Base64 and sending requests to the API.
- **[`olmOCR.ipynb`](olmOCR.ipynb)**: A Jupyter Notebook for testing and developing the OCR extraction model.
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
  - `pypdf`
  - `dspy`
  - `tqdm`
  - `requests`

## Installation

1. Clone the repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_FOLDER>