import streamlit as st
import pandas as pd
import traceback
import dspy
import os
import sys
from utils import process_file, BasicQA

# Dynamically configure the API URL based on command-line args
if len(sys.argv) < 2:
    API_URL = "http://0.0.0.0:8000/api/"
else:
    API_URL = sys.argv[1] + "/api/"

# Define placeholder tokens
huggingface_token = ""
deep_seek_token = ""

# Initialize DSPy Chain
generate_answer = dspy.ChainOfThought(BasicQA)

# Streamlit page setup
st.set_page_config(layout="wide")
st.title("OCR Menu Test")

# Let user choose optimization model
optimization_mode = st.radio(
    "Output optimization mode",
    ["LLama 3 8B", "DeepSeek API"], horizontal=True, index=1,
)

# Optimize the raw OCR result into structured format using LLM

def optimize_menu(parsed_menu):
    if isinstance(parsed_menu, list):
        return pd.DataFrame(parsed_menu)

    try:
        # Choose language model
        if optimization_mode == "LLama 3 8B":
            lm = dspy.LM("huggingface/meta-llama/Meta-Llama-3-8B-Instruct", api_key=huggingface_token, cache=False)
        elif optimization_mode == "DeepSeek API":
            lm = dspy.LM('openai/deepseek-chat', api_key=deep_seek_token, api_base="https://api.deepseek.com", cache=False, max_tokens=8000)

        # Use selected model in context
        with dspy.context(lm=lm):
            list_menu = generate_answer(question=parsed_menu)

        return pd.DataFrame(list_menu["answer"])

    except Exception as e:
        st.error(f"Optimization failed with {optimization_mode}: {e}")
        st.write(traceback.format_exc())
        return pd.DataFrame()

# Display important PDF upload instructions
st.warning(
    "⚠️ If you are uploading a PDF, please make sure to include **only** the pages containing the menu with dishes. "
    "Do not include additional pages such as covers, advertisements, or non-relevant content."
)

# Upload menu file
uploaded_file = st.file_uploader("Upload the menu photo", type=['png', 'jpg', "pdf"])
menu_data = None

# Process uploaded file using external API
with st.spinner("Image processing ..."):
    if uploaded_file is not None:
        data = uploaded_file.read()
        filename, file_extension = os.path.splitext(uploaded_file.name)
        menu_data = process_file(data, file_extension, API_URL)

# Optimize and display processed data
if menu_data:
    with st.spinner("Output optimization ..."):
        dfs = []
        for menu_page in menu_data:
            df = optimize_menu(menu_page)
            if not df.empty:
                dfs.append(df)
        if dfs:
            menu_df = pd.concat(dfs, ignore_index=True)
            st.table(menu_df)
    print("------"*30)