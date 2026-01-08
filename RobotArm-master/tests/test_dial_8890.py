#!/usr/bin/env python3
"""
Test Program: Dial 8890
1. Press green key
2. Dial 8890 (key sequence: 8, 8, 9, 0)
3. Press green key again
4. Press red key (twice)
5. Return to home position

Usage:
  python test_dial_8890.py

Exit:
  - Ctrl+C keyboard interrupt
"""

import time
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotarm.controller import RobotArmController
from robotarm.capture import Capture
from robotarm.utils import get_resource_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    # Initialize controller, load layout and config file
    # Try to load default config
    try:
        config_path = str(get_resource_path('default.yaml', 'configs'))
    except FileNotFoundError:
        config_path = 'configs/default.yaml'

    logger.info(f"Using config: {config_path}")

    controller = RobotArmController(
        config_file=config_path
    )

    # Initialize video capture (try to use 60fps, actual frame rate depends on camera support)
    # Video will be saved as recording/recording_YYYYMMDD_HHMMSS.mp4
    recorder = Capture()

    # Attach capture instance to controller, enable auto-marking
    controller.set_capture(recorder)

    try:
        # Start video recording
        recorder.start()
        time.sleep(0.5)  # Give camera some startup time

        with controller:
            logger.info("Starting dial test: 8890")
            logger.info("Hint: Press Ctrl+C to exit anytime")

            # Step 0: Press red key
            # Red key is configured as mark_on: "lift" in layout, will auto-mark red on lift
            logger.info("[Step 0] Press red key")
            if 'Key_Red' in controller.buttons:
                controller.click_button('Key_Red')
                time.sleep(1)
            else:
                logger.warning("Key_Red not found in layout")

            # Step 1: Press green key
            # Green key is configured as mark_on: "lift" in layout, will auto-mark green on lift
            logger.info("[Step 1] Press green key")
            if 'Key_Green' in controller.buttons:
                controller.click_button('Key_Green')
                time.sleep(1)
            else:
                logger.warning("Key_Green not found in layout")

            # Step 2: Dial 8890
            # Number keys are configured as mark_on: "press" in layout, will auto-mark corresponding color on press
            logger.info("[Step 2] Dial 8890")
            digits = ['8', '8', '9', '0']
            for digit in digits:
                button_name = f'Key_{digit}'
                if button_name in controller.buttons:
                    controller.click_button(button_name)
                    time.sleep(1)
                else:
                    logger.warning(f"{button_name} not found in layout")

            # Step 3: Press green key again (auto-mark)
            logger.info("[Step 3] Press green key again")
            if 'Key_Green' in controller.buttons:
                controller.click_button('Key_Green')
                time.sleep(1)

            # Step 4: Press red key (auto-mark)
            # Red key is configured as mark_on: "lift" in layout, will auto-mark red on lift
            logger.info("[Step 4] Press red key")
            if 'Key_Red' in controller.buttons:
                controller.click_button('Key_Red')
                time.sleep(1)
                controller.click_button('Key_Red')
                time.sleep(1)

            # Step 5: Return to home position
            logger.info("[Step 5] Return to home position")
            controller.lift()  # Ensure Z axis is lifted
            controller.home()  # Return to X0 Y0

            logger.info("Dial test complete!")

    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt (Ctrl+C), exiting...")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
    finally:
        # Stop video recording
        if recorder:
            recorder.stop()

if __name__ == '__main__':
    main()
