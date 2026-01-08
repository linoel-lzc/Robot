#!/usr/bin/env python3
"""
Test script to iterate through all buttons in the layout.
Useful for verifying layout coordinates and calibration accuracy.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotarm import RobotArmController, get_resource_path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Layout Test...")

    try:
        config_path = str(get_resource_path("default.yaml", "configs"))
    except FileNotFoundError:
        config_path = "configs/default.yaml"

    logger.info(f"Using config: {config_path}")

    try:
        with RobotArmController(config_file=config_path) as arm:
            if not arm.buttons:
                logger.error("No buttons loaded! Check layout configuration.")
                return

            # Sort buttons by position for smoother movement
            all_buttons = sorted(
                arm.buttons.keys(),
                key=lambda name: (arm.buttons[name]['center'][1], arm.buttons[name]['center'][0])
            )

            logger.info(f"Found {len(all_buttons)} buttons: {all_buttons}")

            # Click each button
            for btn_name in all_buttons:
                logger.info(f"Clicking {btn_name}...")
                arm.click_button(btn_name)
                time.sleep(0.5)

            logger.info("Returning Home...")
            arm.home()

            logger.info("Test Complete.")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
