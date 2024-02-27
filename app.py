import gradio as gr
import string
from openai import OpenAI
from difflib import Differ
from pydub import AudioSegment

def recognize_speech(api_key, audio):
  if not api_key:
     raise gr.Error("Missing API Key!")

  client = OpenAI(api_key=api_key)
  try:
     print(client.models.list())
  except Exception as e:
      print(str(e))
      raise gr.Error("Invalid API Key.")
  
  if not audio:
    raise gr.Error("Missing record!")
  
  audio_segment = AudioSegment.from_file(audio)
  duration_seconds = len(audio_segment) / 1000
  
  if duration_seconds > 30 :
    raise gr.Error( "Audio input should not be longer than 30 seconds.")
  
  transcript = client.audio.transcriptions.create(model="whisper-1", file=open(audio, "rb"), response_format="text")
  
  return transcript

def improve_speech(api_key, speech):
  if not speech:
     raise gr.Info("Record or type something!")

  if not api_key:
     raise gr.Error("Missing API Key!")

  client = OpenAI(api_key=api_key)
  try:
      print(client.models.list())
  except Exception as e:
      print(str(e))
      raise gr.Error("Invalid API Key.")
  
  prompt = [
     {"role": "assistant", "content": 'You are a english expert. Improve my text. Make it simple, shorter and easy to understand.'}, 
     {"role": "user", "content": speech}
  ]

  response = client.chat.completions.create(model="gpt-3.5-turbo", messages=prompt)
  improved_speech = response.choices[0].message.content
  
  return improved_speech

def diff_texts(text1, text2):
    text1, text2 = remove_punctuation(text1), remove_punctuation(text2)

    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def toggle_main_col(api_key):
  if not api_key:
     return { main_col: gr.Column(visible=False)}

  return { main_col: gr.Column(visible=True)}
   
with gr.Blocks() as ui:
  
  welcome = gr.Markdown(
    """
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
    """)
  
  api_key = gr.Textbox(label="OpenAI API key", placeholder="Enter you key...", lines=1, max_lines=1)
  
  with gr.Column(visible=False) as main_col:
    audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your speech up to 30 sec")
    transcript = gr.Textbox(label="Original Transcript:", lines=4)
    improved_transcript = gr.Textbox(label="Improved Transcript:", lines=4)
    text_diff = gr.HighlightedText(label="Diff", combine_adjacent=True, show_legend=True, color_map={"+": "red", "-": "green"})
    
    recognize_speech_button = gr.Button("Recognize Speech")
    improve_speech_button = gr.Button("Improve Speech")
          
  # events
  api_key.change(toggle_main_col, inputs=[api_key], outputs=[main_col])
  
  recognize_speech_button.click(recognize_speech, inputs=[api_key, audio], outputs=[transcript])
  improve_speech_button.click(improve_speech, inputs=[api_key, transcript], outputs=[improved_transcript])
  
  transcript.change(diff_texts, inputs=[improved_transcript, transcript], outputs=[text_diff])
  improved_transcript.change(diff_texts, inputs=[improved_transcript, transcript], outputs=[text_diff])

ui.queue(max_size=10)
ui.launch()