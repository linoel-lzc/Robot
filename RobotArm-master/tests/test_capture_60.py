#!/usr/bin/env python3
"""
Test Capture class video recording functionality
"""

import sys
import time
import logging
import ffmpeg
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robotarm.capture_60 import Capture, camera4K

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_recording(duration=3):
    """Test basic recording functionality

    Args:
        duration: Recording duration (seconds)
    """
    logger.info("=" * 60)
    logger.info(f"Testing {duration} second video recording")
    logger.info("=" * 60)

    # Create recorder
    capture = Capture(
        camera_name=camera4K,
        width=1920,
        height=1080,
        fps=60,
        output_dir='recording'
    )

    # Start recording
    start_time = datetime.now()
    logger.info(f"Starting {duration} second recording... [{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
    capture.start()

    # Wait and add markers
    time.sleep(1.5)
    # The mark_frame method signature in capture_60.py is (color, message) or similar?
    # Actually capture_60.py's mark_frame signature is (color, label=None) usually.
    # Let's check it or assume it works as used, but update imports and usage.
    # In capture_60.py, it's: def mark_frame(self, color, label=None):
    # Where color can be a string name or tuple.
    capture.mark_frame('red', 'Start marker')

    time.sleep(1.0)
    capture.mark_frame('green', 'Middle marker 1')

    time.sleep(0.5)
    capture.mark_frame('blue', 'Middle marker 2')

    # Wait for remaining time
    remaining = duration - 2.0
    if remaining > 0:
        time.sleep(remaining)

    # Stop recording
    capture.stop()

    # Get output file
    output_file = capture.get_output_file()
    end_time = datetime.now()
    logger.info(f"Recording complete, file saved at: {output_file} [{end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

    # Extract all frames from video
    if output_file and Path(output_file).exists():
        logger.info("=" * 60)
        logger.info("Starting to extract video frames...")

        # Create frame output directory
        frames_dir = Path(output_file).parent / 'frames' / Path(output_file).stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use ffmpeg to extract all frames
            logger.info(f"Extracting video frames to: {frames_dir}")
            (
                ffmpeg
                .input(str(output_file))
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

        logger.info("=" * 60)

    return output_file


def main():
    """Run test"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Capture class recording functionality")
    parser.add_argument(
        '--duration',
        type=int,
        default=4,
        help='Recording duration (seconds), default 4 seconds'
    )

    args = parser.parse_args()

    try:
        test_recording(args.duration)

        logger.info("=" * 60)
        logger.info("Test complete!")
        logger.info("Please check video files in recording directory")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
