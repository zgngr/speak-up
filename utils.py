import string
from difflib import Differ

SPEAK_UP_MARKDOWN = """
    # Speak UP! 
        
    Speak UP! is an interactive web application designed to help users enhance their speech delivery by leveraging the power of Large Language Models (LLMs). 
    The app allows users to record their speech, transcribes the audio to text, then improves the text to be simpler, shorter, and easier to understand. 
    Additionally, it provides a comparison between the original and improved texts, highlighting the changes made.

    ## Features
    - **Audio Recording**: Users can record their speech directly within the app using their device's microphone.
    - **Speech Transcription**: The app transcribes the recorded speech to text using the OpenAI Whisper model.
    - **Text Improvement**: Utilizes OpenAI's GPT models to refine the speech text, focusing on simplicity, brevity, and clarity.
    - **Difference Highlighting**: Displays the differences between the original and improved texts, with changes clearly marked for easy review.

    ## Requirements
    - A valid OpenAI API key.
    - A modern web browser with microphone access.
    """
    
def diff_texts(text1, text2):
    text1, text2 = remove_punctuation(text1), remove_punctuation(text2)
    
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

