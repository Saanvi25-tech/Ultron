# Wake-word / VAD integration helper
# For now we provide two simple modes:
# 1) push-to-talk: user triggers recording manually (frontend button or CLI)
# 2) VAD-based listen: app/vad_listener.listen_and_transcribe() will listen and transcribe when speech is detected

# If you later want always-on wake-word detection with a robust model, integrate Picovoice Porcupine or a similar SDK.

from typing import Literal

Mode = Literal['push-to-talk', 'vad']

DEFAULT_MODE: Mode = 'vad'

def describe():
    return {
        'default_mode': DEFAULT_MODE,
        'options': ['push-to-talk', 'vad'],
        'note': 'Use push-to-talk from the frontend for simpler UX. VAD is provided for a local always-listen style (not recommended for battery-limited devices).'
    }
