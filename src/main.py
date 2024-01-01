"""
Contains the main functions for the Voice Cloning/Synthesis application.
"""
from pathlib import Path
from datasets import Audio, Dataset
import typer
import soundfile as sf

from dataset import create_audio_dataset
from embeddings import prepare_dataset
from tts import synthesise_text
from asr import speech_to_text

app = typer.Typer()


@app.command()
def prepare_embeddings(input_directory: Path, output_directory: Path) -> None:
    """
    Creates a Speaker Embeddings dataset from a directory of audio files.

    Args:
        input_directory (Path): The directory containing the audio files.
        output_directory (Path): The directory to save the Speaker Embeddings dataset to.
    """
    # Creating the Audio Dataset
    dataset = create_audio_dataset(Path(input_directory), None, False)

    # Resampling to 16 kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Creating the Speaker Embeddings
    dataset = dataset.map(prepare_dataset)

    # Saving the dataset
    dataset.save_to_disk(Path(output_directory))


@app.command()
def text_to_speech(text: str, embeddings_dataset: Path, output_file: Path) -> None:
    """
    Synthesises text to speech using the SpeechT5 model with a Speaker Embedding.

    Args:
        text (str): The text to synthesise.
        embeddings_dataset (str): The path to the Speaker Embeddings dataset.
        output_file (str): The path to save the synthesised speech to (.wav).
    """
    # Loading the Speaker Embeddings dataset
    dataset = Dataset.load_from_disk(Path(embeddings_dataset))

    # Prompt the user to select a voice
    styled_text = typer.style(
        f"Select a Voice: 0 - {len(dataset) - 1}", fg=typer.colors.BRIGHT_CYAN
    )
    typer.echo(styled_text)
    styled_text = typer.style("Speaker ID: ", fg=typer.colors.BRIGHT_CYAN)
    speaker = typer.prompt(styled_text, type=int)

    # Selecting a voice
    voice = dataset[speaker]["speaker_embeddings"]

    # Synthesising the speech
    speech = synthesise_text(text, voice)

    # Saving the speech
    sf.write(Path(output_file), speech.numpy(), samplerate=16000)

    styled_text = typer.style(
        f"Saved speech to {output_file}", fg=typer.colors.GREEN
    )
    typer.echo(styled_text)


@app.command()
def speech_to_speech(
    input_file: str, embeddings_dataset: str, output_file: Path
) -> None:
    """
    Synthesises speech to speech using the SpeechT5 model with a Speaker Embedding.

    Args:
        input_file (str): The path to the speech to synthesise (.wav).
        embeddings_dataset (str): The path to the Speaker Embeddings dataset.
        output_file (str): The path to save the synthesised speech to (.wav).
    """
    # Loading the Speaker Embeddings dataset
    dataset = Dataset.load_from_disk(Path(embeddings_dataset))

    # Prompt the user to select a voice
    styled_text = typer.style(
        f"Select a Voice: 0 - {len(dataset) - 1}", fg=typer.colors.BRIGHT_CYAN
    )
    typer.echo(styled_text)
    styled_text = typer.style("Speaker ID: ", fg=typer.colors.BRIGHT_CYAN)
    speaker = typer.prompt(styled_text, type=int)

    # Selecting a voice
    voice = dataset[speaker]["speaker_embeddings"]

    # Converting Speech to Text
    text = speech_to_text(input_file)

    # Printing Transcription
    styled_text = typer.style(f"Transcription: {text}", fg=typer.colors.BRIGHT_CYAN)
    typer.echo(styled_text)

    # Converting Text to Speech
    speech = synthesise_text(text, voice)

    # Saving the speech
    sf.write(Path(output_file), speech.numpy(), samplerate=16000)

    styled_text = typer.style(
        f"Saved speech to {output_file}", fg=typer.colors.GREEN
    )
    typer.echo(styled_text)


if __name__ == "__main__":
    app()

    # Test
    # prepare_embeddings("data/sample", "farnsworth_embeddings")

    # text_to_speech(
    #     "My name is Professor Farnsworth and I smell like an elephant's butt.",
    #     "embeddings/farnsworth_embeddings",
    #     "output/farnsworth_test_1",
    # )
    # speech_to_speech(
    #     "output/farnsworth_test_1.wav",
    #     "embeddings/farnsworth_embeddings",
    #     "output/farnsworth_test_2",
    # )
