import os 
from model import SemanticSearchModel
from utils import load_faiss_index, load_json


class ImageSearch:
  def __init__(self,path='embeddings/',model:SemanticSearchModel=None):
    self.path = path
    self.index= load_faiss_index(os.path.join(path,'index.faiss'))
    self.filenames = load_json(file_path=path,file_name='filenames.json')
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