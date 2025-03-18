import torch
import base64
import json
import logging
from io import BytesIO
from PIL import Image
import dspy
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
class MenuOCRAgent:

    class MenuParsingSignature(dspy.Signature):
        """Format this menu into a list readble for pandas dataframe.
        The datafram should have tre columns: name, ingridients, price, section
        If there is no section, you have to decide the section of each dish among the following categories: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande"""
        question = dspy.InputField(desc="Raw menu input")
        answer: list = dspy.OutputField(desc="should be a list")

    def __init__(self, api_key_token, formatting_model):

        # Set up the OCR model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
        ).eval().to(self.device)
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # It is possibile to set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.    
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        
        # Set up the language model
        if formatting_model == "meta-llama":
             self.lm = dspy.LM("huggingface/meta-llama/Meta-Llama-3-8B-Instruct", api_key=api_key_token, cache=False, max_tokens=15000)
        elif formatting_model == "deepseek":
             self.lm = dspy.LM('openai/deepseek-chat', api_key=api_key_token, api_base="https://api.deepseek.com", cache=False, max_tokens=8000)
        dspy.configure(lm=self.lm)
        self.generate_formatted_menu = dspy.ChainOfThought(self.MenuParsingSignature)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MenuOCRAgent")
        
    def encode_image_file(self, image_path):
        """Encode image file to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            return None

    async def extract_menu_from_image(self, image_path):
        """Extract menu from image using OCR model."""
        image_base64 = self.encode_image_file(image_path)
        if image_base64 is None:
            return ""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract only the menu dishes from the image and provide it as a list\
                            with the following fields: name, price, ingredients, section.\
                            Try to detect the section of each dish among the following categories: Antipasti, Primi, Secondi, Contorni, Dolci, Bevande.\
                            If any of these fields are not present use the value None.\
                            DO NOT PROVIDE ANY OTHER INFORMATION.
                            Example of output:
                            [
                                ["Pizza Margherita", "10.0", "Pomodoro, Mozzarella", "Secondi"],
                                ["Pasta al pomodoro", "8.00", "Pasta, pomodoro, basilico", "Primi"]
                            ]"""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

            # Image processing
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            inputs = self.processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Output generation
            output = self.model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=15000,
                num_return_sequences=1,
                do_sample=True,
            )

            # Output decoding
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

            return text_output[0]
        except Exception as e:
            self.logger.error(f"Error extracting menu: {e}")
            return ""
    
    async def parse_menu(self, text):
        """Parse the menu text and format it into a list."""
        try:
            return self.generate_formatted_menu(question=text)
        except Exception as e:
            self.logger.error(f"Error parsing menu: {e} \n\n Raw menu text: {text}")
            return ""
        
    async def process_menu(self, image_path):
        """Process the image and extract the formatted menu."""

        # Extract menu from image
        menu_text_raw = await self.extract_menu_from_image(image_path)
        if menu_text_raw == "":
            return json.dumps({
            "status": "error",
            "message": "Error extracting menu text",
            "data": None
        })

        # Parse menu
        menu_formatted = await self.parse_menu(menu_text_raw)
        if menu_formatted == "":
            return json.dumps({
            "status": "error",
            "message": "Error formatting menu.",
            "data": None
        })

        return json.dumps({
            "status": "success",
            "message": "The request was processed successfully.",
            "data": menu_formatted
        })
        



