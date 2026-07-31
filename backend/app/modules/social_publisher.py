import logging
from typing import Dict, Any
from app.models.domain import Episode, Universe

logger = logging.getLogger("social_publisher")

class SocialPublisherModule:
    """
    Core Module 12: Social Publisher.
    Formats and prepares episode assets for publishing across YouTube (widescreen), Instagram Reels (9:16 vertical), TikTok, and X.
    """

    def prepare_social_package(self, episode: Episode, universe: Universe) -> Dict[str, Any]:
        title = episode.title
        tags = [f"#{universe.genre.replace(' ', '')}", "#AIFilm", "#Cyberpunk", "#StoryUniverse", "#OpenSourceAI", "#Movie"]
        
        description = f"""
🎬 {title} - {universe.title}
Season {episode.season} Episode {episode.episode_number}

{episode.logline}

🔥 Created autonomously with Open-Source AI (Qwen, FLUX, CogVideoX, Piper, MusicGen).

#AI #Cinema #GenerativeAI {' '.join(tags)}
        """.strip()

        return {
            "episode_id": episode.id,
            "title": title,
            "youtube": {
                "title": f"{universe.title} S{episode.season}E{episode.episode_number}: {title}",
                "description": description,
                "tags": tags,
                "category_id": "1",  # Film & Animation
                "privacy": "public"
            },
            "instagram": {
                "caption": f"{title}\n\n{episode.logline}\n\n{' '.join(tags)}",
                "aspect_ratio": "9:16"
            },
            "tiktok": {
                "caption": f"Episode {episode.episode_number} of {universe.title}: {title}! {' '.join(tags[:3])}",
                "sound": "Original AI Master Score"
            }
        }

social_publisher = SocialPublisherModule()
