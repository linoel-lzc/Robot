#!/usr/bin/env python3
"""
Performer - Automatic Key Sequence Execution Program

Reads sequences/*.yaml config (including config_file reference) and automatically executes key sequences.
"""

import argparse
import datetime
import logging
import time
import yaml
from pathlib import Path

from robotarm.controller import RobotArmController
from robotarm.logging_utils import setup_logging
from robotarm.utils import get_resource_path

logger = logging.getLogger(__name__)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override overwrites base"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_melody(filepath: str) -> dict:
    """Load melody configuration file"""
    try:
        real_path = get_resource_path(filepath, 'scores')
        with open(real_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Melody file not found: {filepath}")
        raise


def create_capture(config: dict, config_path: str, filename_prefix: str, timestamp: str):
    """Create Capture instance based on configuration"""
    capture_config = config.get('capture', {})
    backend = capture_config.get('backend', 'opencv')

    logger.info(f"Using video capture backend: {backend}")

    try:
        if backend == 'pipe':
            # Use ffmpeg pipe mode (capture_pipe.py)
            from robotarm.capture_pipe import Capture
            return Capture(
                config_path=config_path,
                filename_prefix=filename_prefix,
                timestamp=timestamp
            )

        elif backend == 'ffmpeg':
            # Use ffmpeg direct recording mode (capture_60.py)
            from robotarm.capture_60 import Capture
            return Capture(
                camera_name=capture_config.get('camera_name', 'Cisco Desk Camera 4K'),
                width=capture_config.get('width', 1920),
                height=capture_config.get('height', 1080),
                fps=capture_config.get('fps', 60),
                output_dir=capture_config.get('output_dir', 'recording'),
                pixel_format=capture_config.get('pixel_format', 'uyvy422'),
                filename_prefix=filename_prefix,
                timestamp=timestamp
            )

        else:  # Default to opencv (capture.py)
            from robotarm.capture import Capture
            return Capture(
                camera_name=capture_config.get('camera_name', 'Cisco Desk Camera 4K'),
                width=capture_config.get('width', 1920),
                height=capture_config.get('height', 1080),
                fps=capture_config.get('fps', 30),  # opencv default 30fps
                output_dir=capture_config.get('output_dir', 'recording'),
                filename_prefix=filename_prefix,
                timestamp=timestamp
            )

    except Exception as e:
        logger.error(f"Failed to create Capture: {e}")
        return None


def execute_action(controller: RobotArmController, action: dict, index: int):
    """Execute a single action"""
    action_type = action.get('type', 'button')
    comment = action.get('comment', '')
    wait_before = action.get('wait_before', 0)
    wait_after = action.get('wait_after', 0)

    # Wait before execution
    if wait_before > 0:
        logger.debug(f"Waiting {wait_before} seconds...")
        time.sleep(wait_before)

    # Log action
    if comment:
        logger.info(f"[Action {index + 1}] {comment}")
    else:
        logger.info(f"[Action {index + 1}] Type: {action_type}")

    # Execute action based on type
    if action_type == 'button':
        button_name = action.get('name')
        if not button_name:
            logger.error(f"Action {index + 1}: button type missing name parameter")
            return
        controller.click_button(button_name)

    elif action_type == 'home':
        controller.lift()
        controller.home()

    elif action_type == 'lift':
        controller.lift()

    elif action_type == 'wait':
        duration = action.get('duration', 1)
        logger.info(f"Waiting {duration} seconds...")
        time.sleep(duration)

    elif action_type == 'move':
        x = action.get('x', 0)
        y = action.get('y', 0)
        controller.move_to(x, y)

    else:
        logger.warning(f"Action {index + 1}: Unknown type '{action_type}'")

    # Wait after execution
    if wait_after > 0:
        time.sleep(wait_after)


def run_melody(controller: RobotArmController, melody: dict):
    """Execute the entire melody sequence"""
    name = melody.get('name', 'Unnamed Sequence')
    description = melody.get('description', '')
    actions = melody.get('actions', [])

    logger.info(f"Starting sequence: {name}")
    if description:
        logger.info(f"Description: {description.strip()}")
    logger.info(f"Total {len(actions)} actions")
    logger.info("-" * 50)

    for i, action in enumerate(actions):
        execute_action(controller, action, i)

    logger.info("-" * 50)
    logger.info(f"Sequence '{name}' completed!")


def run_performer(args=None):
    setup_logging("Performer")

    parser = argparse.ArgumentParser(
        description='Performer - Automatic Key Sequence Execution Program',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performer.py                                    Use default score
  python performer.py -m dial_8890.yaml                 Specify score file
  python performer.py --no-record                        Disable video recording
        """
    )

    parser.add_argument(
        '-m', '--melody',
        default='dial_8890.yaml',
        help='Score config filename or path (Default: dial_8890.yaml)'
    )

    parser.add_argument(
        '--no-record',
        action='store_true',
        help='Disable video recording'
    )
    
    # Allow passing args list
    if args is not None:
        parsed_args = parser.parse_args(args)
    else:
        parsed_args = parser.parse_args()

    # Load melody config
    logger.info(f"Loading melody config: {parsed_args.melody}")
    try:
        melody = load_melody(parsed_args.melody)
    except FileNotFoundError:
        return 1

    # Generate unified timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get recording prefix (Prefer recording.prefix, then logging.filename_prefix, default 'recording')
    rec_config = melody.get('recording', {})
    log_config = melody.get('logging', {})

    prefix = rec_config.get('prefix') or log_config.get('filename_prefix') or 'recording'

    # Set log filename: logs/{prefix}_{timestamp}_performer.log
    log_filename = f"{prefix}_{timestamp}_performer.log"
    setup_logging("Performer", filename=log_filename)
    logger.info(f"Log file set: {log_filename}")

    # Get config file path from melody
    config_file_name = melody.get('config_file')
    
    # Load base config
    config = {}
    config_path = None
    
    if config_file_name:
        try:
            # First try exact path or local relative path
            config_path = get_resource_path(config_file_name, 'configs')
            logger.info(f"Loading config file: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # Fallback: if filename has path components, try just the filename in resources
            fallback_attempt = False
            if '/' in config_file_name or '\\' in config_file_name:
                filename_only = Path(config_file_name).name
                try:
                    config_path = get_resource_path(filename_only, 'configs')
                    logger.info(f"Fallback: Loading built-in config: {config_path}")
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    fallback_attempt = True
                except FileNotFoundError:
                    pass
            
            if not fallback_attempt:
                logger.error(f"Config file not found: {config_file_name}")
                return 1

    # Merge override config from melody
    melody_config = melody.get('config', {})
    if melody_config:
        logger.info("Merging override config from melody")
        config = deep_merge(config, melody_config)

    # Initialize Controller
    controller = RobotArmController()
    controller.load_config_from_dict(config)

    # Initialize Video Capture (Optional)
    recorder = None
    if not parsed_args.no_record:
        # Default config path if none was loaded
        if not config_path:
             try:
                 config_path = get_resource_path('default.yaml', 'configs')
             except:
                 pass
        
        # Create Capture instance based on config
        recorder = create_capture(
            config=config,
            config_path=str(config_path) if config_path else 'configs/default.yaml',
            filename_prefix=prefix,
            timestamp=timestamp
        )
        if recorder:
            controller.set_capture(recorder)

    try:
        # Start video recording
        if recorder:
            recorder.start()
            time.sleep(0.5)  # Give camera time to start

        with controller:
            logger.info("Hint: Press Ctrl+C to exit anytime")
            run_melody(controller, melody)

    except KeyboardInterrupt:
        logger.info("\nReceived Keyboard Interrupt (Ctrl+C), exiting...")
    finally:
        # Stop video recording
        if recorder:
            recorder.stop()

    return 0


if __name__ == '__main__':
    exit(run_performer())
