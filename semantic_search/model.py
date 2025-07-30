import torch 
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

class SemanticSearchModel(nn.Module):
    def __init__(self,name='openai/clip-vit-large-patch14'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name)
        self.processor =  CLIPProcessor.from_pretrained(name, use_fast=False)
        self.eval()
    
    def forward(self,images,text):
      inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
      outputs = self.model(**inputs)
      logits_per_image = outputs.logits_per_image
      probs = logits_per_image.softmax(dim=1) 
      return probs
    
    def encode_image(self,images):
        if isinstance(images,list):
            images = [images]
        inputs =  self.processor(images=images,return_tensors='pt')
        image_features = self.model.get_image_features(**inputs)
        return image_features
    
    def encode_text(self,text):
        if isinstance(text,list):
            text = [text]
        inputs =  self.processor(text=text,return_tensors='pt')
        text_features = self.model.get_text_features(**inputs)
        return text_features
    


if __name__ == "__main__":
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    image = Image.open(requests.get(url, stream=True).raw)
    text = 'cat'
    model = SemanticSearchModel()
    with torch.no_grad():
        text_embeds = model.encode_text(text)
        image_embeds = model.encode_image(image)
        print(torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)))
    

