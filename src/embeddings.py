"""
Module containing functions to create Speaker Embeddings using the 
pretrained spkrec-xvect-voxceleb model from Speech Brain.
"""

import os
import torch
from speechbrain.pretrained import EncoderClassifier
from datasets import Dataset, Audio

# Loading Speech Embedding Model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"

speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", "spk_model"),
)


# Function to return Speaker Embeddings
def create_speaker_embeddings(audio):
    """Takes audio as an input and outputs a 512 element vector containing the corresponding speaker embedding.
    Args:
        audio: Audio signal as a numpy array
    Returns:
        speaker_embeddings: Speaker Embeddings as a numpy array
    """
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(audio))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


def prepare_dataset(example):
    """
    Creates Speaker Embeddings for an example in the dataset.
    """
    # Accessing the audio in the example
    audio = example["audio"]

    # Creating the speaker embedding for the example
    example["speaker_embeddings"] = create_speaker_embeddings(audio["array"])
    return example


def create_speaker_embeddings_dataset(input_dataset, output_dataset):
    """
    Creates a Speaker Embeddings dataset from a dataset of audio files.

    Args:
        input_dataset (Path): The filepath of the input dataset.
        output_dataset (Path): The filepath of the output dataset.
    """
    # Loading the dataset from input directory
    dataset = Dataset.load_from_disk(input_dataset)

    # Resampling to 16 kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Creating the Speaker Embeddings
    dataset = dataset.map(prepare_dataset)

    # Saving the dataset
    dataset.save_to_disk(output_dataset)


if __name__ == "__main__":
    create_speaker_embeddings_dataset(
        "data/audio_dataset", "data/speaker_embeddings_dataset"
    )
