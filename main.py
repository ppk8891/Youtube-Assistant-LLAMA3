import streamlit as st
from utils.helper import get_embedding_query, split_document_into_chunks
from utils.create_vector_db import get_vector_db
from utils.yt_transcript_loader import get_video_title, get_transcript
from gpt4all import Embed4All
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load embedding model
embedder = Embed4All(os.getcwd()+'/Embedding Models/nomic-embed-text-v1.f16.gguf')

def get_answer(query, context):
    llm_server_url = 'http://localhost:1234/v1'
    api_key = 'not_needed'
    
    template = """
    You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.
    Be precise. You might not find the answer from paragraphs at the beginning. 
    Sometimes the answer might be at the end of the context paragraph.
    
    "{query}".
    
    From this context "{context}.
    
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    llm = ChatOpenAI(base_url=llm_server_url,api_key=api_key,model='LLAMA3',temperature=0.3)

    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "query":query,
        "context":context
    })

st.title("LLAMA3 - Youtube Assistant 🤖")

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
    print("Searching for similar content.")
    # similarity search
    distances, indices = db.search(query_embeddings.reshape(1, -1), 2)
    
    context = []
    for row in indices:
        for i in row:
            context.append(chunks[i].page_content)
            #print(f"Chunk index {i}: {chunks[i]}")
    print("Search Complete")
    
    context  = " ".join(context)
    #print(context)
    print("Context : ", context)
    st.subheader("Answer: ")
    st.write_stream(get_answer(query,context))