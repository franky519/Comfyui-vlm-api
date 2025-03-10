
from .vlm_api import VLMAPINode

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {

    "VLMAPINode": VLMAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {

    "VLMAPINode": "🤖 VLM API",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
