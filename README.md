# 🎙️ GigaAM STT Plugin for Hermes Agent

Локальный плагин для распознавания речи (Speech-to-Text) на базе модели **GigaAM**. Полностью заменяет стандартный Whisper в Hermes Agent, обеспечивая высокую точность, работу без интернета и полную конфиденциальность ваших голосовых данных.

---

## 🚀 Быстрая установка

```bash
# 1. Установка системных зависимостей (ffmpeg для конвертации, portaudio для записи)
sudo apt update && sudo apt install ffmpeg portaudio19-dev

# 2. Установка и активация плагина через менеджер Hermes
hermes plugins install https://github.com/danilakom/hermes-gigaam-stt

# 3. Активация виртуального окружения Hermes
source ~/.hermes/hermes-agent/venv/bin/activate (.fish)

# 4. Установка необходимых Python-библиотек в окружение Hermes
~/.hermes/bin/uv pip install -r ~/.hermes/plugins/hermes-gigaam-stt/requirements.txt

# 5. Редкатирование config.yaml для активации модели. В ~/.hermes/config.yaml добавьте
stt:
  enabled: true
  language: ru
  provider: gigaam