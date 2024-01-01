'''
Launches a Gradio Interface for AI Voice Cloning/Synthesis.
'''

import gradio as gr
import numpy as np
from datasets import Dataset
from pathlib import Path
from tts import synthesise_text
from asr import speech_to_text

# We'll need to convert the synthesied speech to an int16 array as it is expected by Gradio
# We do this by normalising the audio array by the dynamic range of the target dtype(int16) and then converting from the default NumPy dtype(float64) to int16
target_dtype = np.int16
max_range = np.iinfo(target_dtype).max

def text_to_speech(text: str, embeddings_dataset: Path, speaker_id: int) -> (int, np.ndarray):
    """
    Synthesises text to speech using the SpeechT5 model with a Speaker Embedding.

    Args:
        text (str): The text to synthesise.
        embeddings_dataset (str): The path to the Speaker Embeddings dataset.
        speaker_id (int): The ID of the speaker to use.
    """
    # Loading the Speaker Embeddings dataset
    dataset = Dataset.load_from_disk(Path(embeddings_dataset))

    # Selecting a voice
    speaker_embedding = dataset[int(speaker_id)]["speaker_embeddings"]

    synthesised_speech = synthesise_text(text, speaker_embedding)
    synthesised_speech = (synthesised_speech.numpy() * max_range).astype(np.int16)
    
    return 16000, synthesised_speech

def speech_to_speech(input_file: Path, embeddings_dataset: Path, speaker_id: int):
    """
    Synthesises speech to speech using the SpeechT5 model with a Speaker Embedding.

    Args:
        input_file (str): The path to the speech to synthesise (.wav).
        embeddings_dataset (str): The path to the Speaker Embeddings dataset.
        output_file (str): The path to save the synthesised speech to (.wav).
    """
    # Loading the Speaker Embeddings dataset
    dataset = Dataset.load_from_disk(Path(embeddings_dataset))

    # Selecting a voice
    speaker_embedding = dataset[int(speaker_id)]["speaker_embeddings"]

    # Transcribing the speech
    speech = speech_to_text(input_file)

    # Synthesising the speech
    synthesised_speech = synthesise_text(speech, speaker_embedding)
    synthesised_speech = (synthesised_speech.numpy() * max_range).astype(np.int16)
    
    return speech, (16000, synthesised_speech)

embeddings_datasets = list(Path("embeddings").glob("*"))

with gr.Blocks() as demo:
    gr.Markdown("""# Gradio Interface for AI Voice Cloning/Synthesis""")
    
    with gr.Tab(label="Text to Speech"):
        gr.Interface(
            fn=text_to_speech,
            inputs=[
                gr.Textbox(lines=5, label="Text to Synthesise"),
                gr.Dropdown(choices=embeddings_datasets, label="Embeddings Dataset"),
                gr.Textbox(lines=1, label="Speaker ID", value="0"),
            ],
            outputs=gr.Audio(type="numpy", label="Synthesised Speech")
        )
    
    with gr.Tab(label="Speech to Speech"):
        gr.Interface(
            fn=speech_to_speech,
            inputs=[gr.Audio(sources="microphone", type="filepath", label="Speech to Convert"),
                    gr.Dropdown(choices=embeddings_datasets, label="Embeddings Dataset"),
                    gr.Textbox(lines=1, label="Speaker ID", value="0")],
            outputs=[gr.Textbox(lines=5, label="Transcribed Speech"),
                     gr.Audio(type="numpy", label="Synthesised Speech")]
        )
demo.launch(share=False)