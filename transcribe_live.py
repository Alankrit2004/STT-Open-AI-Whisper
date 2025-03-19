import whisper
import pyaudio
import numpy as np
import wave
import time

# Load Whisper model
model = whisper.load_model("small")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best at 16kHz
CHUNK = 1024  # Buffer size

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening... (Press Ctrl+C to stop)")

try:
    while True:
        frames = []
        start_time = time.time()

        # Record 5 seconds of audio
        for _ in range(0, int(RATE / CHUNK * 5)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Convert recorded data to numpy array
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        result = model.transcribe(audio_data)
        print("Transcript:", result["text"])

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
