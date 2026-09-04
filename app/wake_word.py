"""Wake word placeholder.
Implement a lightweight wake-word detector here (Porcupine, Vosk keyword spotting, or an energy/VAD-based trigger).
For initial development we'll use push-to-talk; this file documents options and provides a small VAD-based example.
"""
# Placeholder: for reliability consider Picovoice Porcupine (commercial) or KWT-based models.
# For a simple VAD-triggered push-to-talk style, use webrtcvad to detect speech and then run transcription.

import webrtcvad
import collections
import sys

print('This is a placeholder for wake-word detection. For now use push-to-talk or call the transcribe endpoint.')
