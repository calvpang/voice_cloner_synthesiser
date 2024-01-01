'''
Module for Text to Speech.
'''
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Loading Text to Mel Spectrogram Model
checkpoint = "microsoft/speecht5_tts"

processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

# Loading Mel Spectrogram to Waveform Model
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Moving models to GPU
model.to(device)
vocoder.to(device)

def synthesise_text(text, speaker_embeddings):
    '''
    Synthesises text to speech using the SpeechT5 model.
    Args:
        text (str): The text to synthesise.
        speaker_embeddings (array): The speaker embeddings to use.
    Returns:
        array: The synthesised speech.
    '''
    embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"].to(device), embeddings.to(device), vocoder=vocoder)
    return speech.cpu()
