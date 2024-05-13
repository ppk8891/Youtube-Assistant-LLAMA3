from utils import create_vector_db
import streamlit as st
import textwrap
from utils.helper import get_embedding_query, split_document_into_chunks
from utils.create_vector_db import get_vector_db
from utils.yt_transcript_loader import get_video_title, get_transcript
import requests
import json
from gpt4all import Embed4All
import os

# Load embedding model
embedder = Embed4All(os.getcwd()+'/Embedding Models/nomic-embed-text-v1.f16.gguf')


def get_response_from_query(query, context):
    llm_server_url = 'http://localhost:1234/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "model": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "messages": [
            {"role": "system", "content": "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."},
            {"role": "user", "content": f"""Answer this question {query}. Based on this context {context}. Make the response short.
             If you think there is not enough context to answer the question, reply a single word 'no'."""}
        ],
        "temperature": 0.2,
        "max_tokens": -1,
        "stream": 'true'
    }

    response = requests.post(llm_server_url, headers=headers, data=json.dumps(data))

    #print(str(response.json()['choices'][0]['message']['content'].replace('\n',' ')))
    return response.json()['choices'][0]['message']['content'].replace('\n',' ')

st.title("LLAMA3 - Youtube Assistant")

def get_context(db, query):
    pass

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(label="Input youtube video url", max_chars=150)
        query = st.sidebar.text_area(label="What would you like to know about?",max_chars=150,key="query")
        # Submit URL and query
        submit_button = st.form_submit_button(label="Submit")
        
        
if query and youtube_url:
    video_title = get_video_title(youtube_url)
    print("query : ",query)
    # get embeddings from transcript
    transcript = get_transcript(youtube_url,video_title)
    chunks = split_document_into_chunks(transcript)

    db = get_vector_db(transcript,embedder)
    # get embeddings from query
    query_embeddings = get_embedding_query(query,embedder)
    
    distances, indices = db.search(query_embeddings.reshape(1, -1), 2)
    context = []
    for row in indices:
        for i in row:
            context.append(chunks[i].page_content)
            #print(f"Chunk index {i}: {chunks[i]}")
    print("searched")
    
    context  = " ".join(context)
    #print(context)
    response = get_response_from_query(query,context)
    response = str(response)
    print('type of response : ',type(response))
    print(response)

    st.subheader("Answer: ")
    st.text(textwrap.fill(response,width=80))