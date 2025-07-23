import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import requests
import numpy as np

class SemanticSearchModel:
    def __init__(self,name='openai/clip-vit-base-patch32'):
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.processor =  AutoProcessor.from_pretrained(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_image(self,images):
        if isinstance(images,list):
            images = [images]
        
        return self.processor(images=images,return_tensors='np')
    
    def encode_text(self,text):
        if isinstance(text,list):
            text = [text]
        
        return self.processor(text=text,return_tensors='np')
    


if __name__ == "__main__":
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    image = Image.open(requests.get(url, stream=True).raw)
    text = 'cat'
    model = SemanticSearchModel()
    

