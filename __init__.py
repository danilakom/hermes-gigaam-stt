import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path

try:
    from agent.transcription_provider import TranscriptionProvider
except ImportError:
    from agent.stt_provider import STTProvider as TranscriptionProvider

logger = logging.getLogger(__name__)

class GigaAMProvider(TranscriptionProvider):
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("[GigaAM] Первая транскрибация: загрузка модели в память...")
            self._model = gigaam.load_model("gigaam")
            logger.info("[GigaAM] Модель успешно загружена!")
        return self._model

    @property
    def name(self) -> str:
        return "gigaam"

    @property
    def display_name(self) -> str:
        return "GigaAM (Local)"

    def is_available(self) -> bool:
        # 1. Проверка системного ffmpeg (безопасная, через shutil)
        if not shutil.which("ffmpeg"):
            logger.error("[GigaAM] Не найден системный пакет ffmpeg. Установите его через менеджер пакетов вашей ОС (например, 'apt install ffmpeg' или 'brew install ffmpeg').")
            return False
            
        # 2. Проверка Python-зависимостей
        missing_deps = []
        for dep in ["gigaam", "yaml", "sounddevice", "numpy"]:
            try:
                __import__(dep)
            except ImportError:
                # Корректируем имя для pyyaml
                dep_name = "pyyaml" if dep == "yaml" else dep
                missing_deps.append(dep_name)

        if missing_deps:
            deps_str = " ".join(missing_deps)
            logger.error(
                f"[GigaAM] Отсутствуют необходимые библиотеки: {deps_str}. "
                f"Пожалуйста, выполните в терминале: pip install {deps_str}"
            )
            return False

        return True

    def transcribe(self, file_path: str, *, model=None, language=None, **extra) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
            
        try:
            # Конвертация в формат, который понимает GigaAM (16kHz, Mono, WAV)
            cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "transcript": "", "error": f"FFmpeg error: {result.stderr}", "provider": self.name}
            
            text = self.model.transcribe(tmp_path)
            transcript = text.get("text", "").strip() if isinstance(text, dict) else str(text).strip()
                
            return {"success": True, "transcript": transcript, "provider": self.name}
            
        except Exception as e:
            return {"success": False, "transcript": "", "error": str(e), "provider": self.name}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

def register(ctx):
    ctx.register_transcription_provider(GigaAMProvider())
    logger.info("[GigaAM] Плагин зарегистрирован как STT провайдер.")