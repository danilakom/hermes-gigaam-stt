sudo apt update && sudo apt install ffmpeg portaudio19-dev

hermes plugins install https://github.com/danilakom/hermes-gigaam-stt --enable

source ~/.hermes/hermes-agent/venv/bin/activate (.fish)
uv pip install gigaam pyyaml sounddevice numpy