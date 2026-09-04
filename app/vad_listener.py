"""VAD-based listener that waits until voice is detected and then records a short clip for transcription.
This is a simple helper using `sounddevice` and `webrtcvad`. On Windows, you may need to install PortAudio
and the sounddevice wheel appropriate for your Python version.

Usage:
  from app.vad_listener import listen_and_transcribe
  text = listen_and_transcribe()

This will block until it captures speech, then return transcription text by calling the transcription pipeline.
"""
import webrtcvad
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
from app.speech_pipeline import transcribe_file

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # 10, 20, or 30 ms
NUM_FRAMES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)


def vad_collector(duration=5, threshold_frames=3):
    vad = webrtcvad.Vad(2)  # aggressiveness 0-3
    print('Listening for speech... (speak now)')
    while True:
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio = recording.flatten()
        # Break into frames
        voiced_frames = 0
        for start in range(0, len(audio), NUM_FRAMES):
            end = start + NUM_FRAMES
            frame = audio[start:end]
            if len(frame) < NUM_FRAMES:
                break
            pcm_bytes = frame.tobytes()
            if vad.is_speech(pcm_bytes, SAMPLE_RATE):
                voiced_frames += 1
                if voiced_frames >= threshold_frames:
                    # Save the recording to a temp file and return path
                    tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(tmp_wav.name, recording.astype('int16'), SAMPLE_RATE)
                    return tmp_wav.name
        # If not enough voice frames, loop and listen again


def listen_and_transcribe():
    wav_path = vad_collector()
    try:
        text = transcribe_file(wav_path)
    finally:
        try:
            import os
            os.remove(wav_path)
        except Exception:
            pass
    return text

if __name__ == '__main__':
    print(listen_and_transcribe())
