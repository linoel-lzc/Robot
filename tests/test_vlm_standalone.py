import cv2
import yaml
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotarm.vlm import create_vlm_client
from robotarm.logging_utils import setup_logging
from robotarm.utils import get_resource_path

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_dummy_image():
    """Creates a simple dummy image (black background with a white rectangle)."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
    cv2.putText(img, "Test", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def main():
    setup_logging("TestVLM")

    try:
        config_path = str(get_resource_path('default.yaml', 'configs'))
    except FileNotFoundError:
        config_path = 'configs/default.yaml'

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)

    logger.info("Creating VLM client...")
    vlm_client = create_vlm_client(config)

    if not vlm_client:
        logger.error("Failed to create VLM client. Check configuration.")
        return

    # Test with specific image and prompt
    # Using a dummy image unless a real one is specified
    image_path = "/Users/branchen/Workspace/RobotArm/images/recording_20251218_155813/frame_000117.jpg"
    prompt = "Is there a text box with the number '8' entered in the image?"

    logger.info(f"Loading image from: {image_path}")
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to read image")
            return
    else:
        logger.warning(f"Image not found at {image_path}, falling back to dummy image")
        image = create_dummy_image()
        prompt = "Is there a white rectangle in this image?"

    logger.info(f"Sending request to VLM with prompt: '{prompt}'")

    response = vlm_client.ask(image, prompt)

    logger.info(f"VLM Response: {response}")

    if response:
        print("\n--- VLM Test Result ---")
        print(response)
        print("-----------------------")
    else:
        print("\n--- VLM Test Failed ---")

if __name__ == "__main__":
    main()
