import google.generativeai as genai
from PIL import Image
from typing import List

class GeminiInterface:
    def __init__(self, api_key) -> None:
        genai.configure(api_key=api_key, transport='rest')
        

    def inference(self, pil_image: Image.Image, prompt: str, model: str) -> str:
        try:

            # Fetch the list of supported models from the Gemini API
            supported_models: List[str] = [model_info.name for model_info in genai.list_models()]

            # Validate if the provided model is supported
            if model not in supported_models:
                raise LookupError(f"Model '{model}' is not supported. Available models: {supported_models}")

            # Create a generative model instance
            generative_model = genai.GenerativeModel(model_name=model)

            # Generate content using the model, prompt, and image
            output = generative_model.generate_content([prompt, pil_image]).text

            return output

        except LookupError as e:
            raise e

        except Exception as e:
            raise RuntimeError(f"Failed to generate Gemini output: {e}")