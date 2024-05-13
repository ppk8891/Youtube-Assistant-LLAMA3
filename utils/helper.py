
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from gpt4all import Embed4All
import os

# GPT4All embeddings for transcript
def get_embedding_document(sentence:str,embedder):
    output = embedder.embed(sentence, prefix='search_document',long_text_mode='mean')
    embedding = np.array(output,dtype='float32')
    return embedding

# GPT4All embeddings for search query
def get_embedding_query(sentence,embedder):
    output = embedder.embed(sentence, prefix='search_query',long_text_mode='mean')
    embedding = np.array(output,dtype='float32')
    return embedding

# To split documents to evenly distributed chunks
def split_document_into_chunks(text):
    #
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(text)
    #
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents(pages)
    return chunks
