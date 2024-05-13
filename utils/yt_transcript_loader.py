from langchain_community.document_loaders import YoutubeLoader
from bs4 import BeautifulSoup 
import requests 

def get_video_title(url:str):
    response = requests.get(url) 
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the <title> tag in the head section of the page
    title_tag = soup.find('title')

    # Extract and clean the text of the title
    video_title = title_tag.text.replace(' - YouTube', '').strip() if title_tag else None

    return video_title

def get_transcript(url:str,video_title):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()
        
    # Assuming transcript is a list of documents, and each document has a 'page_content' attribute
    transcript_text = '\n'.join([doc.page_content for doc in transcript])

    """    # Step 3: Save the transcript to a text file
    file_path = "./Transcripts/"+video_title+"_transcript.txt"  # Specify the file path where you want to save the transcript

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(transcript_text)

    print(f"The transcript has been saved to {file_path}.")
    """    
    return transcript_text
    
