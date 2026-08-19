import os
import subprocess
import tempfile
import logging
from pathlib import Path
import yaml

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
        """Проверяет наличие всех зависимостей и выводит понятные инструкции при их отсутствии."""
        missing_deps = []
        
        try:
            import gigaam
        except ImportError:
            missing_deps.append("gigaam")
            
        try:
            import yaml
        except ImportError:
            missing_deps.append("pyyaml")

        # Проверка системного ffmpeg
        has_ffmpeg = subprocess.call(
            ["which", "ffmpeg"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        ) == 0
        if not has_ffmpeg:
            logger.error(
                "[GigaAM] Критическая ошибка: не найден системный пакет 'ffmpeg'. "
                "Без него конвертация аудио невозможна. "
                "Установите его: 'sudo apt install ffmpeg' (Ubuntu) или 'brew install ffmpeg' (macOS)."
            )
            return False

        if missing_deps:
            deps_str = " ".join(missing_deps)
            logger.error(
                f"[GigaAM] Отсутствуют Python-зависимости: {deps_str}. "
                f"Активируйте виртуальное окружение Hermes и выполните: "
                f"'pip install {deps_str}' (или 'pip install -r requirements.txt' в папке плагина)."
            )
            return False

        return True

    def transcribe(self, file_path: str, *, model=None, language=None, **extra) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
            
        try:
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

def auto_activate_stt():
    marker = Path("~/.hermes/.gigaam_activated").expanduser()
    if marker.exists():
        return

    config_path = Path("~/.hermes/config.yaml").expanduser()
    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
        if "stt" not in config:
            config["stt"] = {}
            
        if config["stt"].get("provider") != "gigaam" or config["stt"].get("enabled") is False:
            config["stt"]["provider"] = "gigaam"
            config["stt"]["enabled"] = True
            
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info("[GigaAM] ✅ STT автоматически активирован в config.yaml")
            
        marker.touch()
    except Exception as e:
        logger.error(f"[GigaAM] Ошибка автоактивации: {e}")

def register(ctx):
    auto_activate_stt()
    ctx.register_transcription_provider(GigaAMProvider())
    logger.info("[GigaAM] Плагин зарегистрирован как STT провайдер.")