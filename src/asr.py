""" Module for Automatic Speech Recognition using Whisper. """
import torch
from transformers import pipeline

# Loading the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)


def speech_to_text(audio_file) -> str:
    """
    Converts an audio file to text.

    Args:
        audio_file (str): The path to the audio file.

    Returns:
        str: The text from the audio file.
    """
    outputs = pipe(audio_file, max_new_tokens=256)
    return outputs["text"]
