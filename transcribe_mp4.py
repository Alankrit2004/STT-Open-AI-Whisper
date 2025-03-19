import whisper
import ffmpeg
import sys

def transcribe_mp4(file_path):
    print("Loading whisper model....")
    model = whisper.load_model("small")

    print("Extracting audio from video....")
    audio_path = "temp_audio.wav"

    ffmpeg.input(file_path).output(audio_path).run(overwrite_output=True, quiet=True)

    print("Transcribing audio....")
    result = model.transcribe(audio_path)

    print("\nTranscript:")
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_mp4.py <video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    transcribe_mp4(file_path)