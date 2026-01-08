import base64
import cv2
import logging
import numpy as np
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class VLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, system_prompt: Optional[str] = None, temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 300, top_k: Optional[int] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"VLMClient initialized with model: {model} at {base_url} (temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, top_k={top_k})")

    def encode_image(self, image: np.ndarray) -> str:
        """Encodes an OpenCV image (numpy array) to base64 string."""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image to JPEG")
        return base64.b64encode(buffer).decode('utf-8')

    def ask(self, image: np.ndarray, prompt: str) -> Optional[str]:
        """Sends an image and a prompt to the VLM and returns the response."""
        try:
            base64_image = self.encode_image(image)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            })

            # Prepare extra parameters
            extra_body = {}
            if self.top_k is not None:
                extra_body['top_k'] = self.top_k

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=extra_body if extra_body else None,
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            logger.error(f"Error calling VLM: {e}")
            return None

def create_vlm_client(config: dict) -> Optional[VLMClient]:
    """Creates a VLMClient instance from configuration.

    Environment variables (override config file):
        VLM_BASE_URL: VLM service base URL
        VLM_API_KEY: API key for authentication
    """
    import os

    vlm_config = config.get('vlm')
    if not vlm_config:
        logger.warning("No VLM configuration found.")
        return None

    # Prioritize environment variables, then use configuration file
    base_url = os.environ.get('VLM_BASE_URL') or vlm_config.get('base_url')
    api_key = os.environ.get('VLM_API_KEY') or vlm_config.get('api_key', 'sk-no-key-required')
    model = vlm_config.get('model')
    system_prompt = vlm_config.get('system_prompt')
    temperature = vlm_config.get('temperature', 0.7)
    top_p = vlm_config.get('top_p', 1.0)
    max_tokens = vlm_config.get('max_tokens', 300)
    top_k = vlm_config.get('top_k')

    if not base_url or not model:
        logger.warning("Incomplete VLM configuration: base_url and model are required.")
        return None

    return VLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        top_k=top_k
    )
