"""ComfyUI Director - a timeline for building long videos from Qwen-Image keyframes and Wan clips."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from server import PromptServer
    from .director.routes import register

    register(PromptServer.instance.routes)
except Exception as e:  # running outside ComfyUI (tests)
    print(f"[director] routes not registered: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
