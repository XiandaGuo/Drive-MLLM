from PIL import Image
import torch
import copy
import warnings
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

class LlavaInterface:

    def __init__(self, pretrained, model_name, conv_template) -> None:
        warnings.filterwarnings("ignore")
        self.device = "cuda"
        self.device_map = "auto"
        self.conv_template = conv_template
        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map)
        self.model.eval()
    
    def inference(self, pil_image: Image.Image, prompt: str, max_tokens = 4096) -> str:
        image = pil_image
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        
        # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        question = DEFAULT_IMAGE_TOKEN + prompt
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]


        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]