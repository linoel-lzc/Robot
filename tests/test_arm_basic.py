#!/usr/bin/env python3
"""
Basic Robot Arm Test Program - Reference Original Driver Code
Uses automatic serial port detection and performs basic movement tests
"""

import time
import logging
import serial.tools.list_ports
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def list_available_ports():
    """List all available serial ports"""
    plist = list(serial.tools.list_ports.comports())
    logger.info(f"Found {len(plist)} serial ports")

    if len(plist) <= 0:
        logger.error("No ports found!")
        return []

    for idx, port in enumerate(plist):
        logger.info(f"  [{idx}] {port.device} - {port.description}")

    return plist

def test_with_robotarm():
    """Test using RobotArmController with config file"""
    from robotarm import RobotArmController, get_resource_path

    logger.info("=" * 60)
    logger.info("Testing with RobotArmController")
    logger.info("=" * 60)

    try:
        config_path = str(get_resource_path("default.yaml", "configs"))
    except FileNotFoundError:
        config_path = "configs/default.yaml"

    logger.info(f"Using config: {config_path}")

    try:
        with RobotArmController(config_file=config_path) as arm:
            logger.info("Connected successfully!")
            arm.move_z(0)
            time.sleep(0.5)

            # Click calibration buttons to verify calibration
            test_buttons = ["Key_Green", "Key_Red", "Key_0"]
            for btn_name in test_buttons:
                if btn_name in arm.buttons:
                    logger.info(f"Clicking {btn_name}...")
                    arm.click_button(btn_name)
                    time.sleep(0.5)
                else:
                    logger.warning(f"Button {btn_name} not found in layout")

            # Return to origin
            logger.info("Returning to origin...")
            arm.home()
            time.sleep(0.5)

            logger.info("Test complete!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

    return True

def test_with_raw_serial(port_index=None):
    """Test using raw pyserial (reference original driver code)"""
    import serial

    logger.info("=" * 60)
    logger.info("Testing with raw Serial")
    logger.info("=" * 60)

    plist = list_available_ports()

    if not plist:
        return False

    # If no port specified, try auto-select
    if port_index is None:
        # On macOS, look for /dev/cu.* ports
        import platform
        if platform.system() == "Darwin":
            for idx, port in enumerate(plist):
                if port.device.startswith('/dev/cu.'):
                    port_index = idx
                    logger.info(f"Auto-selected macOS serial port: [{idx}] {port.device}")
                    break

        # If still not selected, use the first one
        if port_index is None:
            port_index = 0
            logger.info(f"Using first port: [{port_index}] {plist[port_index].device}")

    if port_index >= len(plist):
        logger.error(f"Port index {port_index} out of range (0-{len(plist)-1})")
        return False

    serial_name = plist[port_index].device
    logger.info(f"Connecting to port: {serial_name}")

    try:
        serialFd = serial.Serial(serial_name, 93450, timeout=60)
        logger.info(f"Port name: {serialFd.name}")

        # Important: Wait for robot to be ready
        logger.info("Waiting for robot to be ready (2 seconds)...")
        time.sleep(2)

        # Move to test position
        logger.info("Moving to position (X85, Y96)...")
        # Update command format to G-code style if controller expects it
        serialFd.write("G1X85Y96\r".encode())
        time.sleep(0.5)

        # Perform 3 press tests
        for i in range(3):
            logger.info(f"Cycle {i+1}")

            logger.info("  Pressing down (Z10)...")
            serialFd.write("G1Z10\r".encode())
            time.sleep(2)

            logger.info("  Lifting up (Z0)...")
            serialFd.write("G1Z0\r".encode())
            time.sleep(2)

        # Return to origin
        logger.info("Returning to origin (X0, Y0)...")
        serialFd.write("G1X0Y0\r".encode())
        time.sleep(0.5)

        serialFd.close()
        logger.info("Test complete!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Robot Arm Basic Test')
    parser.add_argument('--list', action='store_true', help='List all available serial ports')
    parser.add_argument('--raw', action='store_true', help='Use raw Serial test')
    parser.add_argument('--port', type=int, help='Specify serial port index (use with --raw)')

    args = parser.parse_args()

    if args.list:
        list_available_ports()
        return

    if args.raw:
        test_with_raw_serial(args.port)
    else:
        test_with_robotarm()

if __name__ == "__main__":
    main()
