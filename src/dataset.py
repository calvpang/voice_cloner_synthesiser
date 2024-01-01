'''
Module for creating an Audio dataset from a directory of audio files.
'''
from pathlib import Path
from datasets import Audio, Dataset

def create_audio_dataset(input_directory: str, output_directory: str, save: bool) -> None:
    '''
    Creates an Audio dataset from a directory of audio files.

    Args:
        input_directory (Path): The directory containing the audio files.
        output_directory (Path): The directory to save the audio dataset to.
    '''
    # Retrieve the filepaths in the input directory
    filepaths = [str(x) for x in Path(input_directory).glob("*")]

    # Creating an audio dataset from the filepaths
    audio_dataset = Dataset.from_dict({"audio": filepaths}).cast_column("audio", Audio())

    # Saving the audio dataset
    if save:
        audio_dataset.save_to_disk(output_directory)
    else:
        return audio_dataset

if __name__ == "__main__":
    create_audio_dataset("data/audio", "data/audio_dataset", save=True)
