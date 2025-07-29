import numpy as np
import faiss
import json 
import os 
from model import SemanticSearchModel


class ImageSearch:
  def __init__(self,path='embeddings/',model:SemanticSearchModel=None):
    self.path = path
    self.index= faiss.read_index(os.path.join(path,'index.faiss'))
    with open(os.path.join(path, 'filenames.json'), 'r') as f:
      self.filenames = json.load(f)
    self.model = model or SemanticSearchModel()

  def search(self,query,top_k=5,threshold=50):
    query_embedding = self.model.encode_text(query)
    query_embedding = query_embedding.detach().cpu().numpy().astype('float32').reshape(1, -1)
    D, I = self.index.search(query_embedding, top_k)
    results = [
            (self.filenames[i], float(score))
            for i, score in zip(I[0], D[0])
            if score >= threshold
        ]
    return results


if __name__ == "__main__":
  print(ImageSearch(path='embeddings').search('Image of a dog?'))