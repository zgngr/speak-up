import shutil
import gradio as gr
from openai import OpenAI
from pydub import AudioSegment
from difflib import Differ

prompt = [{"role": "assistant", "content": 'You are a english expert. Improve my text. Make it simple, shorter and easy to understand.'}]

def transcribe(api_key, audio):

    api_key = "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"

    ##### start input checks
    # if not api_key:
    #     raise gr.Error("OpenAI API Key required.")
    
    audio_segment = AudioSegment.from_file(audio)
    duration_seconds = len(audio_segment) / 1000
    if duration_seconds > 20:
        raise gr.Error("Audio input should not be longer than 20 seconds.")
    ##### end input checks

    file_path = "to_transcribe.wav"
    shutil.copyfile(audio, file_path)

    #### OpenAI calls
    client = OpenAI(api_key=api_key)
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    prompt.append({"role": "user", "content": transcript})
    response = client.chat.completions.create(model="gpt-4", messages=prompt)
    gpt = response.choices[0].message.content

    return transcript, gpt

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

with gr.Blocks() as ui:
    welcome =gr.Markdown(
    """
    # Speak UP!
    A toy LLM app for helping you to level up your speeches.
    """)

    input_api_key = gr.Text(label="OpenAI API Key", placeholder="Enter your OpenAI API Key here")
    input_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your speech", max_length=20)
    output_original_transcript = gr.Text(label="Original", visible=False)
    output_improved_transcript = gr.Text(label="Improved", visible=False)

    speak_up_button = gr.Button("Improve my speech")

    speak_up_button.click(
        transcribe, 
        inputs=[input_api_key, input_audio], 
        outputs=[output_original_transcript, output_improved_transcript]
    )


# ui = gr.Interface(
#     fn=transcribe, 
#     inputs=[input_api_key, input_audio], 
#     outputs=[output_original_transcript, output_improved_transcript]
# )


if __name__ == "__main__":
    ui.launch()
