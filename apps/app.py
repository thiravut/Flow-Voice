"""
app.py — Entry point for Multi-Language TTS Voice Cloning Web UI.

Usage:
    python app.py
    python app.py --port 7861
    python app.py --share
"""

import argparse
import logging

from config import HISTORY_DIR, SERVER_HOST, SERVER_PORT, UPLOADS_DIR
from emotions import EmotionManager
from engine import EngineRouter
from ui import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Language TTS Voice Cloning")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--host", type=str, default=SERVER_HOST, help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    # Ensure runtime directories exist
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize components
    router = EngineRouter()
    emotion_manager = EmotionManager()

    # Build and launch UI
    demo = create_ui(router, emotion_manager)
    logger.info("Launching on http://%s:%d", args.host, args.port)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
