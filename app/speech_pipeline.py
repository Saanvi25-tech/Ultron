"""Simple speech utilities for Ultron.
Uses faster-whisper if available for local CPU/GPU transcription.
"""
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_file(path: str, model_size: str = 'small') -> str:
    model = WhisperModel(model_size, device='cpu')
    segments, info = model.transcribe(path)
    text = []
    for segment in segments:
        text.append(segment.text)
    return " ".join(text)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python app/speech_pipeline.py audio.wav')
    else:
        print(transcribe_file(sys.argv[1]))
