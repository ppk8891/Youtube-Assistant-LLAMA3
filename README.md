# Youtube-Assistant-LLAMA3 🤖

### Overview
Youtube-Assistant-LLAMA3 is a simple RAG (Retrieval-Augmented Generation) chatbot designed to answer questions about YouTube videos.
By providing a YouTube URL, you can ask the bot questions, and it will generate responses based on the content of the video.

### Set-up Instructions
#### Prerequisites
LLM Inference Server: You need to set up an LLM inference server using LM Studio.

#### Steps
Start the LLM Inference Server:

1. Make sure you have LM Studio installed and configured.
2. Start the server with your local LLAMA3 8B model.

Run the Streamlit App:

``` streamlit run main.py ```

* You need at least 16 GB of RAM with 4GB of VRAM for partial GPU offload.
