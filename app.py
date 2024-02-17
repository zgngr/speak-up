import shutil
import gradio as gr
from openai import OpenAI
from pydub import AudioSegment
from difflib import Differ

def speech_to_text(client, speech):
    transcript = client.audio.transcriptions.create(model="whisper-1", file=speech, response_format="text")
    return transcript

def improve_speech(client, speech):
    prompt = [
        {"role": "assistant", "content": 'You are a english expert. Improve my text. Make it simple, shorter and easy to understand.'}, 
        {"role": "user", "content": speech}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=prompt)
    improved_speech = response.choices[0].message.content
    return improved_speech

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

with gr.Blocks() as ui:
    with gr.Column():
        welcome =gr.Markdown(
        """
        # Speak UP!
        A toy LLM app for helping you to level up your speeches.
        """)

    with gr.Column():
        api_key = gr.Textbox(label="OpenAI API key", placeholder="Enter you key...", lines=1, max_lines=1)
        audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your speech", max_length=20)
        run_btn = gr.Button("Speak UP!")
    
    with gr.Column(visible=False) as results:
        text_original_transcript = gr.Text(label="Original transcript:", lines=3)
        text_improved_transcript = gr.Text(label="Improved transcript:", lines=3)
        text_diff = gr.HighlightedText(label="Diff", combine_adjacent=True, show_legend=True, color_map={"+": "red", "-": "green"})

    def run(api_key, speech):
        #### input validation
        if not api_key:
            raise gr.Error("Missing API Key!")

        if not speech:
            raise gr.Error("Record something!")

        #### saving speech
        file_path = "speech.wav"
        shutil.copyfile(speech, file_path)

        audio_segment = AudioSegment.from_file(file_path)
        duration_seconds = len(audio_segment) / 1000
        if duration_seconds > 20:
            raise gr.Error("Audio input should not be longer than 20 seconds.")
    
        gr.Info("Starting process...")
        #### OpenAI calls
        client = OpenAI(api_key=api_key)
        original_transcript = speech_to_text(client, open(file_path, "rb"))
        improved_transcript = improve_speech(client, original_transcript)
    
        #### diff
        diff = diff_texts(original_transcript, improved_transcript)

        return {
            results: gr.Column(visible=True),
            text_original_transcript: original_transcript, 
            text_improved_transcript: improved_transcript, 
            text_diff: diff
        }

    def show_components():
        return {
            results: gr.Column(visible=True),
            text_original_transcript: "", 
            text_improved_transcript: "", 
            text_diff: ""
        }

    def hide_components():
        return {
            results: gr.Column(visible=False),
            text_original_transcript: "", 
            text_improved_transcript: "",
            text_diff: ""
        }
    
    #### events

    run_btn.click(run, inputs=[api_key, audio], outputs=[results, text_original_transcript, text_improved_transcript, text_diff])

    audio.clear(hide_components, None, outputs=[results, text_original_transcript, text_improved_transcript, text_diff])

if __name__ == "__main__":
    ui.launch()