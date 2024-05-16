import requests
import numpy as np
import faiss
from utils.helper import get_embedding_document,split_document_into_chunks

d = 768  # number of embedding dimension

def get_vector_db(transcript,embedder):
    
    chunks = split_document_into_chunks(transcript)
    print("Number of Chunks : ",len(chunks))
    chunk_0 = str(chunks[0]).split('=')[1]

    index = faiss.IndexFlatL2(d)
    X = np.zeros((len(chunks),d),dtype='float32')
    sentences = []
    print("Creating embeddings for the yt transcript.")
    for i, chunk in enumerate(chunks):
        
        #print(f"creating embedding for chunk {i}")
        sentence = str(chunks[i]).split('=')[1]
        #print(sentence)
        sentences.append(sentence)
        #create embedding
        embedding = get_embedding_document(sentence,embedder)
        embedding = embedding.reshape(1, -1)

        #add embedding to faiss index
        index.add(embedding)
    print("Vector DB created")        
    return index