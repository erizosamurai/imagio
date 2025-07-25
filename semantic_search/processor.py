import os 
from PIL import Image
import numpy as np 
import torch
from torch.utils.data import DataLoader

from model import SemanticSearchModel

class ImageDataset:
  def __init__(self,path):
    self.path = path
    self.images = [ f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self,idx):
    image_path = os.path.join(self.path,self.images[idx])
    image = Image.open(image_path).convert('RGB')
    return image, self.images[idx]
  
class ImageProcessor:
  def __init__(self,path,model:SemanticSearchModel=None, batch_size=32):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.path = path
      self.model = model or SemanticSearchModel()
      self.batch_size = batch_size

  def process_folder(self):
      dataset = ImageDataset(self.path)
      dataloader = DataLoader(
              dataset,
              batch_size= self.batch_size,
              shuffle=False,
              num_workers=0,
              collate_fn=lambda batch: list(zip(*batch)),  # To separate imgs, filenames
          )
      
      all_embeddings = []
      file_names = []

      print(f"Processing folder: {self.path}")
      print(f"Using device: {self.device}")

      with torch.no_grad():
          for imgs_batch,filename_batch in dataloader:
              embeddings = self.model.encode_image(imgs_batch)

              all_embeddings.append(embeddings.cpu())
              file_names.extend(filename_batch)

      embeddings_tensor = torch.cat(all_embeddings)
      return embeddings_tensor.numpy(), file_names

  def save_embeddings(self,embeddings,file_names,save_path='embeddings/'):
      os.makedirs(save_path,exist_ok=True)
      np.save(os.path.join(save_path,'embeddings.npy'),embeddings)
      np.save(os.path.join(save_path,'filenames.npy'),file_names)
  
  def load_embeddings(self):
      pass


if __name__ == "__main__":
   process_images = ImageProcessor(path='Data')
   embeddings, file_name = process_images.process_folder()
   process_images.save_embeddings(embeddings,file_name)
   a = np.load('embeddings/filenames.npy')
   print(a)
    