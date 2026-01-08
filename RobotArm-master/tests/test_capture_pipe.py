#!/usr/bin/env python3
"""
Test capture_pipe module video recording functionality

Test contents:
1. Load parameters from config file
2. Real-time frame marking functionality
3. Video frame extraction and verification
"""

import sys
import time
import logging
import ffmpeg
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robotarm.capture_pipe import Capture, create_capture_from_config
from robotarm.utils import get_resource_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_recording_with_config(duration=4):
    """Test recording functionality with parameters loaded from config file

    Args:
        duration: Recording duration (seconds)
    """
    logger.info("=" * 60)
    logger.info(f"Test 1: Load parameters from config file, record {duration} second video")
    logger.info("=" * 60)

    # Create recorder from config file
    try:
        config_path = str(get_resource_path('default.yaml', 'configs'))
    except FileNotFoundError:
        config_path = 'configs/default.yaml'

    capture = Capture(
        config_path=config_path,
        filename_prefix="test_pipe"
    )

    # Start recording
    start_time = datetime.now()
    logger.info(f"Starting {duration} second recording... [{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
    capture.start()

    # Wait and add markers (consistent with capture.py, marker applied to next frame)
    time.sleep(1.5)
    capture.mark_frame('red', 'Mark_1')
    logger.info("Marker set: red - Mark_1")

    time.sleep(1.0)
    capture.mark_frame('green', 'Mark_2')
    logger.info("Marker set: green - Mark_2")

    time.sleep(0.5)
    capture.mark_frame('blue', 'Mark_3')
    logger.info("Marker set: blue - Mark_3")

    # Wait for remaining time
    remaining = duration - 3.0
    if remaining > 0:
        time.sleep(remaining)

    # Stop recording
    capture.stop()

    # Get recording statistics
    stats = capture.get_recording_stats()
    output_file = stats['output_file']
    end_time = datetime.now()

    logger.info(f"Recording complete, file saved at: {output_file}")
    logger.info(f"Total frames: {stats['frame_count']}")
    if stats['first_frame_time'] and stats['last_frame_time']:
        duration_actual = (stats['last_frame_time'] - stats['first_frame_time']).total_seconds()
        fps_actual = stats['frame_count'] / duration_actual if duration_actual > 0 else 0
        logger.info(f"Actual duration: {duration_actual:.2f}s, actual frame rate: {fps_actual:.2f}fps")

    # Extract video frames
    if output_file and Path(output_file).exists():
        extract_frames(output_file)

    return output_file


def test_recording_with_overrides(duration=3):
    """Test recording functionality with override parameters

    Args:
        duration: Recording duration (seconds)
    """
    logger.info("=" * 60)
    logger.info(f"Test 2: Use override parameters, record {duration} second video (30fps)")
    logger.info("=" * 60)

    try:
        config_path = str(get_resource_path('default.yaml', 'configs'))
    except FileNotFoundError:
        config_path = 'configs/default.yaml'

    # Use factory function to create, and override fps parameter
    capture = create_capture_from_config(
        config_path=config_path,
        filename_prefix="test_pipe_30fps",
        fps=30  # Override fps from config file
    )

    # Start recording
    start_time = datetime.now()
    logger.info(f"Starting {duration} second recording... [{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
    capture.start()

    # Wait and add markers (using BGR tuples)
    time.sleep(1.0)
    capture.mark_frame((0, 255, 255), 'Yellow_BGR')  # Yellow
    logger.info("Marker set: (0,255,255) - Yellow_BGR")

    time.sleep(1.0)
    capture.mark_frame([255, 0, 255], 'Magenta_BGR')  # Magenta
    logger.info("Marker set: [255,0,255] - Magenta_BGR")

    # Wait for remaining time
    remaining = duration - 2.0
    if remaining > 0:
        time.sleep(remaining)

    # Stop recording
    capture.stop()

    # Get recording statistics
    stats = capture.get_recording_stats()
    output_file = stats['output_file']

    logger.info(f"Recording complete, file saved at: {output_file}")
    logger.info(f"Total frames: {stats['frame_count']}")

    return output_file


def test_rapid_markers(duration=3):
    """Test rapid consecutive markers

    Args:
        duration: Recording duration (seconds)
    """
    logger.info("=" * 60)
    logger.info(f"Test 3: Rapid consecutive marker test, record {duration} seconds")
    logger.info("=" * 60)

    try:
        config_path = str(get_resource_path('default.yaml', 'configs'))
    except FileNotFoundError:
        config_path = 'configs/default.yaml'

    capture = Capture(
        config_path=config_path,
        filename_prefix="test_pipe_rapid"
    )

    capture.start()
    time.sleep(0.5)  # Wait for recording to stabilize

    # Rapid consecutive markers
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    for i, color in enumerate(colors):
        capture.mark_frame(color, f'Rapid_{i}')
        logger.info(f"Rapid marker {i}: {color}")
        time.sleep(0.2)  # 200ms interval

    # Wait for remaining time
    remaining = duration - 0.5 - len(colors) * 0.2
    if remaining > 0:
        time.sleep(remaining)

    capture.stop()

    stats = capture.get_recording_stats()
    logger.info(f"Recording complete, total frames: {stats['frame_count']}")
    logger.info(f"Marked {len(colors)} frames")

    return stats['output_file']


def extract_frames(video_path: str):
    """Extract video frames

    Args:
        video_path: Video file path
    """
    logger.info("-" * 40)
    logger.info("Starting to extract video frames...")

    # Create frame output directory
    frames_dir = Path(video_path).parent / 'frames' / Path(video_path).stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use ffmpeg to extract all frames
        logger.info(f"Extracting video frames to: {frames_dir}")
        (
            ffmpeg
            .input(str(video_path))
            .output(str(frames_dir / 'frame_%06d.jpg'),
                   format='image2',
                   vcodec='mjpeg',
                   qscale=2)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        # Count extracted frames
        frame_files = list(frames_dir.glob('frame_*.jpg'))
        logger.info(f"Frame extraction complete, extracted {len(frame_files)} frames")
        logger.info(f"Output directory: {frames_dir}")

    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        logger.error(f"Error extracting frames:\n{stderr}")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}", exc_info=True)

    logger.info("-" * 40)


def main():
    """Run tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test capture_pipe module recording functionality")
    parser.add_argument(
        '--duration',
        type=int,
        default=4,
        help='Recording duration (seconds), default 4 seconds'
    )
    parser.add_argument(
        '--test',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Run specific test (1: config file, 2: override parameters, 3: rapid markers)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )

    args = parser.parse_args()

    try:
        if args.all:
            # Run all tests
            test_recording_with_config(args.duration)
            logger.info("\n")
            test_recording_with_overrides(args.duration)
            logger.info("\n")
            test_rapid_markers(args.duration)
        elif args.test == 1:
            test_recording_with_config(args.duration)
        elif args.test == 2:
            test_recording_with_overrides(args.duration)
        elif args.test == 3:
            test_rapid_markers(args.duration)
        else:
            # Default run test 1
            test_recording_with_config(args.duration)

        logger.info("=" * 60)
        logger.info("Test complete!")
        logger.info("Please check video files in recording directory")
        logger.info("Extracted frames are saved in recording/frames/ directory")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
