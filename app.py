import gradio as gr
from openai import OpenAI
from datetime import datetime
from pydub import AudioSegment
from functools import lru_cache
from utils import diff_texts, remove_punctuation, SPEAK_UP_MARKDOWN
from prompts import SEMANTIC_ZOOM_PROMPT, SPEECH_IMPROVEMENT_PROMPT, EXTRACT_WIZDOM_PROMPT

@lru_cache(maxsize=100)
def improve_speech(api_key, text):
  if not text:
     raise gr.Info("Record or type something!")

  client = get_openai_client(api_key)
  
  system_message = {"role": "system", "content": SPEECH_IMPROVEMENT_PROMPT}
  user_message = {"role": "user", "content": text}
  messages = [system_message, user_message]

  response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
      
  return response.choices[0].message.content

@lru_cache(maxsize=100)
def summarize_text(api_key, text, zoom_level):
  if not text:
     raise gr.Info("Type something!")

  client = get_openai_client(api_key)
  
  system_message = {"role": "system", "content": SEMANTIC_ZOOM_PROMPT}
  user_message = {"role": "user", "content": "SUMMARY LEVEL: " + str(zoom_level) + "\n" + text}
  messages = [system_message, user_message]

  response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    
  return response.choices[0].message.content

@lru_cache(maxsize=100)
def extract_wizdom(api_key, text):
  if not text:
     raise gr.Info("Type something!")

  client = get_openai_client(api_key)
  
  system_message = {"role": "system", "content": EXTRACT_WIZDOM_PROMPT}
  user_message = {"role": "user", "content": text}
  messages = [system_message, user_message]
  
  response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    
  return response.choices[0].message.content

def recognize_speech(api_key, audio):
  client = get_openai_client(api_key)
  
  if not audio:
    raise gr.Error("Missing record!")
  
  audio_segment = AudioSegment.from_file(audio)
  duration_seconds = len(audio_segment) / 1000
  
  if duration_seconds > 30 :
    raise gr.Error( "Audio input should not be longer than 30 seconds.")
  
  transcript = client.audio.transcriptions.create(model="whisper-1", file=open(audio, "rb"), response_format="text")
  
  return transcript

def get_openai_client(api_key):
  if not api_key:
     raise gr.Error("Missing API Key!")

  client = OpenAI(api_key=api_key)
  try:
    client.models.list()
  except Exception as e:
      print(str(e))
      raise gr.Error("Invalid API Key.")
  
  return client
 
########################## UI
LEVELS = {
  1 :"Level 1: Single Sentence Summary",
  2 :"Level 2: Twitter Post Summary (40-60 words)",
  3 :"Level 3: Executive Summary (80-100 words)",
  4 :"Level 4: High-Level Overview (suitable for decision-making)",
  5 :"Level 5: Detailed Summary (100-120 words, covering all main points and supporting arguments)"
}

def toggle_main_col(api_key):
  if not api_key:
     return { main_col: gr.Column(visible=False)}

  return { main_col: gr.Column(visible=True)}
    
def zoom_level_changed(zoom_level):
  return gr.Slider(1, 5, value=zoom_level, label=LEVELS[zoom_level], info="", interactive=True, step=1)
  
with gr.Blocks() as ui:
  
  api_key = gr.Textbox(label="OpenAI API key", placeholder="Enter you key...", lines=1, max_lines=1)
  
  with gr.Tabs(visible=False) as main_col:

    with gr.TabItem("Speech Enhancement"):
      with gr.Column():
        audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your speech up to 30 sec")
        transcript = gr.Textbox(label="Original Transcript:", lines=4)
        improved_transcript = gr.Textbox(label="Improved Transcript:", lines=4)
        text_diff = gr.HighlightedText(label="Diff", combine_adjacent=True, show_legend=True, color_map={"+": "red", "-": "green"})
        recognize_speech_button = gr.Button("Recognize Speech")
        improve_speech_button = gr.Button("Improve Speech")

    with gr.TabItem("Semantic Zoom"):
      with gr.Column():
        input_text = gr.Textbox(label="Original Text:", lines=4)
        zoom_level = gr.Slider(1, 5, value=1, label=LEVELS[1], interactive=True, step=1)
        semantic_zoom_button = gr.Button("Summarize")
        zoomed_text = gr.Markdown(label="Zoomed Text:")
        
    with gr.TabItem("Extract Wizdom"):
      with gr.Column():
        input_text_2 = gr.Textbox(label="Original Text:", lines=4)
        extract_wizdom_button = gr.Button("Wizdomize")
        wizdom_text = gr.Markdown(label="Zoomed Text:")
          
  # events
  api_key.change(toggle_main_col, inputs=[api_key], outputs=[main_col])
  
  recognize_speech_button.click(recognize_speech, inputs=[api_key, audio], outputs=[transcript])
  improve_speech_button.click(improve_speech, inputs=[api_key, transcript], outputs=[improved_transcript])
  
  transcript.change(diff_texts, inputs=[improved_transcript, transcript], outputs=[text_diff])
  improved_transcript.change(diff_texts, inputs=[improved_transcript, transcript], outputs=[text_diff])
  
  semantic_zoom_button.click(summarize_text, inputs=[api_key, input_text, zoom_level], outputs=[zoomed_text])
  zoom_level.change(zoom_level_changed, inputs=[zoom_level], outputs=[zoom_level])
  
  extract_wizdom_button.click(extract_wizdom, inputs=[api_key, input_text_2], outputs=[wizdom_text])

ui.queue(max_size=10)
ui.launch()