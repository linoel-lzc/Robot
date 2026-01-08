#!/usr/bin/env python3
"""
Test script to verify 3-point calibration by clicking all calibration buttons.
"""

import time
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotarm import RobotArmController, get_resource_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting 3-Point Calibration Test...")

    try:
        config_path = str(get_resource_path("default.yaml", "configs"))
    except FileNotFoundError:
        config_path = "configs/default.yaml"

    logger.info(f"Using config: {config_path}")

    try:
        with RobotArmController(config_file=config_path) as arm:
            # Get calibration points directly from controller's loaded config
            calib = arm.config.get('calibration', {})
            buttons_to_test = []

            for point_key in ['point1', 'point2', 'point3']:
                if point_key in calib:
                    buttons_to_test.append(calib[point_key]['name'])

            if len(buttons_to_test) != 3:
                logger.error(f"Expected 3 calibration points, found {len(buttons_to_test)}")
                return

            logger.info(f"Calibration buttons: {buttons_to_test}")

            # Click each calibration button
            for btn_name in buttons_to_test:
                if btn_name in arm.buttons:
                    logger.info(f"Clicking {btn_name}...")
                    arm.click_button(btn_name)
                    time.sleep(1)
                else:
                    logger.error(f"Button {btn_name} not found in layout!")

            logger.info("Returning Home...")
            arm.home()

            logger.info("Test Complete.")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
