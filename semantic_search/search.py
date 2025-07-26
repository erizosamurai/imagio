import torch
import os 
from model import SemanticSearchModel

class ImageSearch:
  def __init__(self,path='embeddings/',model:SemanticSearchModel=None):
    self.path = path
    self.embeddings = torch.load(os.path.join(path,'embeddings.pt'),weights_only=False)
    self.filenames = torch.load(os.path.join(path,'filenames.pt'),weights_only=False)
    self.model = model or SemanticSearchModel()
  
  def search(self,query,top_k=3):
    query_embedding = self.model.encode_text(query)
    similarity = torch.matmul(query_embedding,self.embeddings.T).squeeze(0)
    top_k_indices = similarity.topk(top_k).indices.tolist()
    file_names = [self.filenames[i] for i in top_k_indices]
    return file_names


if __name__ == "__main__":
  print(ImageSearch(path='embeddings').search('Image of a dog?'))