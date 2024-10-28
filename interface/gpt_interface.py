from PIL import Image
from openai import OpenAI
from typing import List
from utils import pil2base64

class GptInterface:
    
    def __init__(self, api_key) -> None:
        self.client = OpenAI(api_key=api_key)
    
    
    def inference(self, pil_image: Image.Image, prompt: str, model: str, max_tokens: int = 300) -> str:

        try:
            # Fetch the list of supported models from the OpenAI API
            supported_models: List[str] = [model_info.id for model_info in self.client.models.list().data]

            # Validate if the provided model is supported
            if model not in supported_models:
                raise LookupError(f"Model '{model}' is not supported. Available models: {supported_models}")

            # Convert the PIL image to a base64-encoded string
            image_base64 = pil2base64(pil_image)
            image_url = f"data:image/jpeg;base64,{image_base64}"

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )

            # Extract and return the output from the API response
            output = response.choices[0].message.content
            return output

        except LookupError as e:
            raise e

        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI output: {e}")