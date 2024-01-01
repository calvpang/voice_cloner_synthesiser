import librosa
import soundfile as sf
from pydub import AudioSegment

# Load the audio files
audio_1 = AudioSegment.from_mp3("data/sample/farnsworth_1.mp3")
audio_2 = AudioSegment.from_mp3("data/sample/farnsworth_2.mp3")
audio_3 = AudioSegment.from_mp3("data/sample/farnsworth_3.mp3")
audio_4 = AudioSegment.from_mp3("data/sample/farnsworth_4.mp3")
audio_5 = AudioSegment.from_mp3("data/sample/farnsworth_5.mp3")
audio_6 = AudioSegment.from_mp3("data/sample/farnsworth_6.mp3")

combined = audio_1 + audio_2 + audio_3 + audio_4 + audio_5 + audio_6

# Export combined audio to a temporary file
combined.export("temp.wav", format="wav")

# Load the audio with librosa
y, sr = librosa.load("temp.wav", sr=None)

# Slow down the audio by 0.8
y_slow = librosa.effects.time_stretch(y=y, rate=1)

# Save the slowed down audio
sf.write("data/sample/farnsworth_sample.wav", y_slow, sr)