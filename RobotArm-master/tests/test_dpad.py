#!/usr/bin/env python3
"""
Test script for D-Pad (direction keys) functionality.
Tests Up, Down, Left, Right, and Center buttons.
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
    logger.info("Starting D-Pad Test...")

    try:
        config_path = str(get_resource_path("default.yaml", "configs"))
    except FileNotFoundError:
        config_path = "configs/default.yaml"

    logger.info(f"Using config: {config_path}")

    # D-Pad buttons to test (in order: center, up, right, down, left)
    dpad_buttons = ["D_Pad_Center", "D_Pad_Up", "D_Pad_Right", "D_Pad_Down", "D_Pad_Left"]

    try:
        with RobotArmController(config_file=config_path) as arm:
            if not arm.buttons:
                logger.error("No buttons loaded! Check layout configuration.")
                return

            # Verify D-Pad buttons exist
            missing = [btn for btn in dpad_buttons if btn not in arm.buttons]
            if missing:
                logger.error(f"Missing D-Pad buttons in layout: {missing}")
                return

            logger.info(f"Testing D-Pad buttons: {dpad_buttons}")

            # Test each D-Pad button
            for btn_name in dpad_buttons:
                logger.info(f"Clicking {btn_name}...")
                arm.click_button(btn_name)
                time.sleep(0.8)

            # Test pattern: Up -> Down -> Left -> Right -> Center
            logger.info("\n=== Testing D-Pad Pattern ===")
            pattern = ["D_Pad_Up", "D_Pad_Down", "D_Pad_Left", "D_Pad_Right", "D_Pad_Center"]
            for btn_name in pattern:
                logger.info(f"  {btn_name}")
                arm.click_button(btn_name)
                time.sleep(0.5)

            logger.info("Returning Home...")
            arm.home()

            logger.info("D-Pad Test Complete.")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

