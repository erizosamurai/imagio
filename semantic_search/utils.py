import os
import faiss 
import json 

def save_faiss_index(index,file_path):
    faiss.write_index(index,file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def save_json(file,file_name,file_path):
   with open(os.path.join(file_path, file_name), 'w') as f:
        json.dump(file, f, indent=4)

def load_json(file_path,file_name):
    with open(os.path.join(file_path, file_name), 'r') as f:
       filenames = json.load(f)
    return filenames
