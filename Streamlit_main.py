import streamlit as st
import pandas as pd
import traceback
import requests
import base64
import ast
import re
import dspy
import os
import sys
from utils import process_file, BasicQA

# API configuration
#API_URL = "https://7d55-34-142-179-95.ngrok-free.app/api/"
if len(sys.argv) < 2:
    API_URL = "http://0.0.0.0:8000" + "/api/"
else:
    API_URL = sys.argv[1] + "/api/"


huggingface_token = ""
deep_seek_token = ""

# Initialize DSPy module
generate_answer = dspy.ChainOfThought(BasicQA)

# Set the title of the Streamlit app
st.set_page_config(layout="wide")

st.title("OCR Menu Test")

# Radio button to select optimization mode
optimization_mode = st.radio(
    "Output optimization mode",
    ["LLama 3 8B", "DeepSeek API"], horizontal=True, index=1,
)


# Main function to handle menu optimization
def optimize_menu(parsed_menu):
    if isinstance(parsed_menu, list):  # If the parsed menu is already a list, display it as a table
        return pd.DataFrame(parsed_menu)
    # Otherwise, optimize based on the selected mode
    try:
        if optimization_mode == "LLama 3 8B":
            lm = dspy.LM("huggingface/meta-llama/Meta-Llama-3-8B-Instruct", api_key=huggingface_token, cache=False)
        elif optimization_mode == "DeepSeek API":
            lm = dspy.LM('openai/deepseek-chat', api_key=deep_seek_token, api_base="https://api.deepseek.com", cache=False, max_tokens=8000)
            
        #dspy.configure(lm=lm)
        with dspy.context(lm=lm):
            list_menu = generate_answer(question=parsed_menu)
        return pd.DataFrame(list_menu["answer"])
    except Exception as e:
        st.error(f"Optimization failed with {optimization_mode}: {e}")
        st.write(traceback.format_exc())
        return pd.DataFrame()

st.warning(
    "⚠️ If you are uploading a PDF, please make sure to include **only** the pages containing the menu with dishes. "
    "Do not include additional pages such as covers, advertisements, or non-relevant content."
)

# File uploader widget for uploading an image
uploaded_file = st.file_uploader("Upload the menu photo", type=['png', 'jpg', "pdf"])
menu_data = None

# Process the uploaded image and display results
with st.spinner("Image processing ..."):
    if uploaded_file is not None:
        data = uploaded_file.read()
        filename, file_extension = os.path.splitext(uploaded_file.name)
        menu_data = process_file(data, file_extension, API_URL)

if menu_data:
    with st.spinner("Output optimization ..."):
        dfs = []
        for menu_page in menu_data:
            print("RAW INPUT", menu_page)
            df = optimize_menu(menu_page)
            print("OPTIMIZED OUTPUT", df)
            if not df.empty:
                dfs.append(df)
        if dfs:
            print("CONCAT")
            menu_df = pd.concat(dfs, ignore_index=True)
            st.table(menu_df)
    print("------"*30)
