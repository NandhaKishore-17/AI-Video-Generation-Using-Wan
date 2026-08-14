from abc import ABC, abstractmethod


class VideoEngine(ABC):
    """Abstract interface for video rendering engines."""

    @abstractmethod
    def render_placeholder(self, job_id: str) -> str:
        """Generate a placeholder MP4 for the given job.
        Returns the absolute path to the generated video file.
        """
        pass

    @abstractmethod
    def render_video(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        """Generate a video clip for the given scene prompt and return its path."""
        pass

    @abstractmethod
    async def render_video_async(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        """Async variant for use by the compatibility wrapper."""
        pass
