import os
from PIL import Image
import torch
import faiss
from torch.utils.data import DataLoader

from semantic_search.model import SemanticSearchModel
from semantic_search.utils import save_faiss_index, save_json, get_image_files
class ImageDataset:
  def __init__(self,path):
    self.path = path
    self.images = get_image_files(self.path)
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self,idx):
    image_path = os.path.join(self.path,self.images[idx])
    image = Image.open(image_path).convert('RGB')
    return image, self.images[idx]
  
class ImageProcessor:
  def __init__(self,path,model:SemanticSearchModel=None, batch_size=32,dimension=768):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.path = path
    self.model = model or SemanticSearchModel()
    self.batch_size = batch_size
    self.index = faiss.IndexFlatIP(dimension)

  def process_folder(self,workers=0):
    dataset = ImageDataset(self.path)
    dataloader = DataLoader(
            dataset,
            batch_size= self.batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=lambda batch: list(zip(*batch)),  # To separate imgs, filenames
        )

    all_embeddings = []
    file_names = []

    print(f"Processing folder: {self.path}")
    print(f"Using device: {self.device}")

    with torch.no_grad():
      for imgs_batch,filename_batch in dataloader:
        embeddings = self.model.encode_image(imgs_batch).to(self.device)

        all_embeddings.append(embeddings.cpu())
        file_names.extend(filename_batch)

    embeddings_tensor = torch.cat(all_embeddings)

    return embeddings_tensor.detach().cpu().numpy(), file_names

  def save_embeddings(self,embeddings,file,save_path='embeddings/'):
      os.makedirs(save_path,exist_ok=True)
      self.index.add(embeddings)
      save_faiss_index(self.index, os.path.join(save_path, 'index.faiss'))
      save_json(file=file,file_name='filenames.json',file_path=save_path)
  

if __name__ == "__main__":
   process_images = ImageProcessor(path='Data')
   embeddings, file_name = process_images.process_folder()
   process_images.save_embeddings(embeddings,file_name)
   a = faiss.read_index('embeddings/index.faiss')
   print(a.ntotal)
    