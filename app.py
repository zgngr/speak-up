import os, shutil, subprocess
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe(audio):
    file_path = "to_transcribe.wav"
    shutil.copyfile(audio, file_path)

    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")

    subprocess.call(["say", transcript])

    return transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(sources=["microphone"], type="filepath"), outputs="text")

ui.launch()