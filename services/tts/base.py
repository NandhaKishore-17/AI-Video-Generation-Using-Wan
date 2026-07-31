from abc import ABC, abstractmethod

class BaseTTSBackend(ABC):
    """
    Abstract base interface for modular Text-to-Speech backends.
    """

    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        pass
