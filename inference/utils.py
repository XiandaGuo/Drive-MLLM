from PIL import Image
import io
import base64
from typing import Callable, Any
import time


def pil2base64(pil_image: Image.Image, save_format) -> str:
    """
    Converts a PIL image object to a base64-encoded string.

    Parameters:
    pil_image (PIL.Image.Image): The image to be converted to base64.

    Returns:
    str: The base64-encoded string representation of the image.
    """
    try:
        binary_stream = io.BytesIO()
        pil_image.save(binary_stream, format=save_format)
        binary_data = binary_stream.getvalue()
        return base64.b64encode(binary_data).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")


def inference_with_retry(
        inference: Callable,
        *args: Any,
        max_retries: int = 3,
        delay: int = 3,
        **kwargs: Any
) -> str:
    """
    Executes an inference function with automatic retries on failure.

    Args:
        inference (Callable[..., str]): Inference function to call.
        *args (Any): Positional arguments for the inference function.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        delay (int, optional): Time delay (seconds) between retry attempts. Defaults to 3 seconds.
        **kwargs (Any): Keyword arguments for the inference function.

    Returns:
        str: Output from the inference function on success.

    Raises:
        Exception: If the maximum number of retries is exceeded without success.
    """
    retries: int = 0
    while retries < max_retries:
        try:
            output_data: str = inference(*args, **kwargs)
            return output_data
        except Exception as e:
            print(f"Error calling inference: {e}. Retrying {retries + 1}/{max_retries}...")
            retries += 1
            time.sleep(delay)

    raise Exception(f"Failed to complete inference after {max_retries} retries.")