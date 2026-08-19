import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
import warnings

# Подавляем известные предупреждения PyTorch, исходящие из внутренней реализации библиотеки gigaam
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

try:
    from agent.transcription_provider import TranscriptionProvider
except ImportError:
    from agent.stt_provider import STTProvider as TranscriptionProvider

logger = logging.getLogger(__name__)

# Максимальная длительность сегмента в секундах (лимит GigaAM)
MAX_SEGMENT_DURATION = 20


def get_audio_duration(file_path: str) -> float:
    """Получает длительность аудиофайла в секундах через ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
        logger.warning(f"[GigaAM] Не удалось получить длительность аудио: {e}")
    return 0.0


def split_audio(file_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Разбивает аудиофайл на сегмент через ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(file_path),
        "-ss", str(start_time),
        "-t", str(duration),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


class GigaAMProvider(TranscriptionProvider):
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import gigaam
            logger.info("[GigaAM] Первая транскрибация: загрузка модели в память...")
            self._model = gigaam.load_model("rnnt")
            logger.info("[GigaAM] Модель успешно загружена!")
        return self._model

    @property
    def name(self) -> str:
        return "gigaam"

    @property
    def display_name(self) -> str:
        return "GigaAM (Local)"

    def is_available(self) -> bool:
        # Проверка ffmpeg
        if not shutil.which("ffmpeg"):
            print("\033[91m[GigaAM] ❌ ОШИБКА: Не найден системный пакет ffmpeg.\033[0m")
            print("\033[93m[Подсказка] Установите: apt install ffmpeg\033[0m\n")
            return False
        
        # Проверка ffprobe (нужен для определения длительности)
        if not shutil.which("ffprobe"):
            print("\033[91m[GigaAM]  ОШИБКА: Не найден системный пакет ffprobe.\033[0m")
            print("\033[93m[Подсказка] Установите: apt install ffmpeg (ffprobe входит в пакет)\033[0m\n")
            return False
        
        # Проверка Python-зависимостей
        missing_deps = []
        for dep in ["gigaam", "yaml", "sounddevice", "numpy"]:
            try:
                __import__(dep)
            except ImportError:
                dep_name = "pyyaml" if dep == "yaml" else dep
                missing_deps.append(dep_name)
        
        if missing_deps:
            deps_str = " ".join(missing_deps)
            print(f"\033[91m[GigaAM]  ОШИБКА: Отсутствуют библиотеки: {deps_str}\033[0m")
            print(f"\033[93m[Подсказка] Выполните: pip install {deps_str}\033[0m\n")
            return False
        
        return True

    def transcribe(self, file_path: str, *, model=None, language=None, **extra) -> dict:
        """
        Транскрибирует аудиофайл. Если файл длиннее 20 секунд, 
        автоматически разбивает его на сегменты и транскрибирует каждый отдельно.
        """
        # Получаем длительность аудио
        duration = get_audio_duration(file_path)
        
        if duration <= 0:
            # Если не удалось получить длительность, пробуем транскрибировать как есть
            return self._transcribe_single_segment(file_path)
        
        if duration <= MAX_SEGMENT_DURATION:
            # Короткий файл - транскрибируем целиком
            return self._transcribe_single_segment(file_path)
        
        # Длинный файл - разбиваем на сегменты
        return self._transcribe_long_audio(file_path, duration)

    def _transcribe_single_segment(self, file_path: str) -> dict:
        """Транскрибирует один аудиофайл (до 20 секунд)."""
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

    def _transcribe_long_audio(self, file_path: str, total_duration: float) -> dict:
        """
        Транскрибирует длинный аудиофайл, разбивая его на сегменты по 20 секунд.
        """
        num_segments = int(total_duration / MAX_SEGMENT_DURATION) + (1 if total_duration % MAX_SEGMENT_DURATION > 0 else 0)
        
        logger.info(f"[GigaAM] Аудио длинное ({total_duration:.1f} сек). Разбиваю на {num_segments} сегментов...")
        
        transcripts = []
        temp_files = []
        
        try:
            for i in range(num_segments):
                start_time = i * MAX_SEGMENT_DURATION
                # Для последнего сегмента берём оставшееся время
                segment_duration = min(MAX_SEGMENT_DURATION, total_duration - start_time)
                
                # Создаём временный файл для сегмента
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_segment:
                    segment_path = tmp_segment.name
                    temp_files.append(segment_path)
                
                # Разбиваем аудио на сегмент
                if not split_audio(file_path, start_time, segment_duration, segment_path):
                    return {
                        "success": False, 
                        "transcript": "", 
                        "error": f"Ошибка разбиения аудио на сегмент {i+1}/{num_segments}",
                        "provider": self.name
                    }
                
                # Транскрибируем сегмент
                result = self._transcribe_single_segment(segment_path)
                
                if not result["success"]:
                    return {
                        "success": False,
                        "transcript": " ".join(transcripts),  # Возвращаем то, что успели
                        "error": f"Ошибка транскрибации сегмента {i+1}/{num_segments}: {result['error']}",
                        "provider": self.name
                    }
                
                transcripts.append(result["transcript"])
                logger.info(f"[GigaAM] Сегмент {i+1}/{num_segments} транскрибирован")
            
            # Склеиваем все транскрипции
            full_transcript = " ".join(transcripts)
            
            return {
                "success": True,
                "transcript": full_transcript,
                "provider": self.name
            }
            
        except Exception as e:
            return {
                "success": False,
                "transcript": " ".join(transcripts),
                "error": f"Ошибка при обработке длинного аудио: {str(e)}",
                "provider": self.name
            }
        finally:
            # Удаляем все временные файлы сегментов
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)


def register(ctx):
    ctx.register_transcription_provider(GigaAMProvider())
    logger.info("[GigaAM] Плагин зарегистрирован как STT провайдер.")