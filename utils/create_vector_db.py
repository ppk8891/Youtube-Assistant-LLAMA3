import requests
import numpy as np
import faiss
from utils.helper import get_embedding_document,split_document_into_chunks

d = 768  # number of embedding dimension

def get_vector_db(transcript,embedder):
    """
    with open (transcript_file,'r',encoding='utf-8') as file:
        transcript = file.read()
    """    
    chunks = split_document_into_chunks(transcript)
    print("Docs : ",len(chunks))
    chunk_0 = str(chunks[0]).split('=')[1]

    index = faiss.IndexFlatL2(d)
    X = np.zeros((len(chunks),d),dtype='float32')
    sentences = []
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
            
    return index