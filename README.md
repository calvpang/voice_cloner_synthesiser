# Voice Synthesiser

Inspired by the "Device that makes anyone sound like Hubert J. Farnsworth".

![Device that makes anyone sound like Hubert J. Farnsworth](image.png)

This repository contains everything you need to synthesise your own voice for others to use against you!
Once trained, you can use Text-to-Speech or Speech-to-Speech.

## Pre-requisites
To begin, record a few examples of your speech and save them in the `data/audio` directory. At present .wav & .mp3 are supported file formats.
It is recommended to have longer recordings displaying use of a larger vocabluary and different phonemes.

## Environment Setup
Simply install [miniforge](https://github.com/conda-forge/miniforge), navigate to this repository and run the following command in Terminal.
```
conda env create -f environment.yml
```

## Usage
### CLI
Typer has been used to develop the CLI. You can view the help documentation as follows.
```
python src/main.py --help
```

To begin, you will need to create speech embeddings from your pre-recorded speeches.
```
python src/main.py prepare-embeddings data/sample embeddings/farnsworth
```

Once the speech embeddings are created, you can synthesise text to speech using your speech embeddings.
```
python src/main.py text-to-speech "Good news everyone! I'm a horse's butt!" embeddings/farnsworth output/farnsworth_sample_1.wav
```

Alternatively, you can upload another audio file and it will convert the speech using your speech embeddings.
```
python src/main.py speech-to-speech output/farnsworth_sample_1.wav embeddings/farnsworth output/farnsworth_sample_2.wav
```

### Gradio
After creating your speech embeddings using the CLI, you can simply run the following.
```
python src/webui.py
```

**NOTE: You will be prompted for a Speaker ID. This is an index depending on the number of audio files you have saved in your input directory.**