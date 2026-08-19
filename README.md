# 🎙️ GigaAM STT Plugin for Hermes Agent

Локальный плагин для распознавания речи (Speech-to-Text) на базе модели **GigaAM**. Полностью заменяет стандартный Whisper в Hermes Agent.

---

## 🚀 Быстрая установка

```bash
# 1. Установка системных зависимостей (ffmpeg для конвертации, portaudio для записи)
#sudo apt update && sudo apt install ffmpeg portaudio19-dev

# 2. Установка и активация плагина через менеджер Hermes
#hermes plugins install https://github.com/danilakom/hermes-gigaam-stt

# 3. Активация виртуального окружения Hermes
#source ~/.hermes/hermes-agent/venv/bin/activate (.fish)

# 4. Установка необходимых Python-библиотек в окружение Hermes
#uv pip install gigaam pyyaml sounddevice numpy