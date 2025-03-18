import asyncio
import logging

from src.ocr_agent.Latest_Version_MrT import AnalystAgent
from src.celery.config import HUGGINGFACE_TOKEN, DEEPSEEK_TOKEN
from src.celery.worker import app

analyst_agent = MenuOCRAgent(
    formatting_model="meta-llama",
    api_key_token=HUGGINGFACE_TOKEN,
)

@app.task
def add(x, y):
    return x + y

@app.task
def process_menu_task(image_path: str):
    """
    Celery task to process an image and extract the formatted menu using the process_menu method.
    
    Parameters:
        image_path (str): The path to the image to be processed.
    
    Returns:
        str: JSON string containing the result of the process_menu execution.
    """
    logging.getLogger("process_menu_task")

    result = asyncio.run(analyst_agent.process_menu(image_path))

    return result
