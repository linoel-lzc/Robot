#!/usr/bin/env python3
"""
Video Capture Module - Pipe Mode
Provides high frame rate video capture and recording functionality, supports running in independent threads.
Uses ffmpeg pipe mode for high-performance video recording + real-time frame marking.

Features:
- Reads all parameters from YAML config file
- Uses ffmpeg pipe mode for frame-by-frame processing
- Supports real-time frame marking (fixed 1-frame delay)
- Thread-safe, can call mark_frame() during robot arm operations
"""

import logging
import threading
import subprocess
import platform
import numpy as np
import cv2
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load full configuration from config file

    Args:
        config_path: Config file path

    Returns:
        Full configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_colors_from_layout(layout_path: str) -> dict:
    """Load color mapping from layout file

    Args:
        layout_path: Layout config file path

    Returns:
        Dictionary mapping color name -> BGR value
    """
    with open(layout_path, 'r', encoding='utf-8') as f:
        layout = yaml.safe_load(f)
    return layout.get('device', {}).get('colors', {})


class Capture:
    """Video capture class for recording high frame rate video, uses ffmpeg pipe mode for real-time frame marking"""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        layout_path: Optional[str] = None,
        filename_prefix: str = "recording",
        timestamp: Optional[str] = None,
        **overrides
    ):
        """
        Initialize video recorder.

        Args:
            config_path: Config file path, reads capture config and layout_file path from it
            layout_path: Layout file path (optional, if not provided reads from config_path's layout_file field)
            filename_prefix: Video filename prefix
            timestamp: Optional timestamp string for filename
            **overrides: Override parameters from config file, e.g. width=1920, fps=30
        """
        # Load full config
        config = load_config(config_path)
        capture_config = config.get('capture', {})

        # Apply defaults and config
        self.camera_name = overrides.get('camera_name', capture_config.get('camera_name', 'Cisco Desk Camera 4K'))
        self.width = overrides.get('width', capture_config.get('width', 1920))
        self.height = overrides.get('height', capture_config.get('height', 1080))
        self.fps = overrides.get('fps', capture_config.get('fps', 60))
        self.pixel_format = overrides.get('pixel_format', capture_config.get('pixel_format', 'uyvy422'))
        self.output_dir = Path(overrides.get('output_dir', capture_config.get('output_dir', 'recording')))

        self.filename_prefix = filename_prefix
        self.timestamp = timestamp
        self.is_recording = False
        self.thread = None
        self.input_process = None
        self.output_process = None
        self.output_file = None

        # Recording statistics
        self.first_frame_time = None
        self.last_frame_time = None
        self.frame_count = 0

        # Frame marking - thread-safe (consistent with capture.py)
        self.marker_lock = threading.Lock()
        self.pending_marker = None  # (color, label)

        # Marker region config (prefer capture config, fallback to video_processing)
        marker_config = capture_config.get('marker_region') or config.get('video_processing', {}).get('marker_region', {})
        self.marker_region = {
            'x': marker_config.get('x', 10),
            'y': marker_config.get('y', 10),
            'width': marker_config.get('width', 100),
            'height': marker_config.get('height', 100)
        }

        # Color mapping: color name -> BGR value
        # Prefer passed layout_path, otherwise read from config's layout_file field
        effective_layout_path = layout_path or config.get('layout_file')
        self.color_map = {}

        if effective_layout_path:
            try:
                self.color_map = load_colors_from_layout(effective_layout_path)
                logger.info(f"Loaded {len(self.color_map)} colors from {effective_layout_path}")
            except Exception as e:
                logger.warning(f"Cannot load colors from layout: {e}")

        if not self.color_map:
            logger.warning("No color config loaded, frame marking will use passed BGR values")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Capture initialized: {self.camera_name} @ {self.width}x{self.height} {self.fps}fps")

    def set_marker_region(self, region: dict):
        """Set marker region configuration

        Args:
            region: Dictionary containing x, y, width, height
        """
        if not isinstance(region, dict):
            return
        self.marker_region.update(region)
        logger.info(f"Marker region updated: {self.marker_region}")

    def mark_frame(self, color: Union[str, tuple, list], label: str = ""):
        """Mark the next frame - thread-safe, can be called during robot arm operations

        Consistent with capture.py behavior: sets pending_marker, mark will be applied to next processed frame.

        Args:
            color: Mark color, can be:
                   - Color name string, e.g. 'red', 'green'
                   - BGR tuple/list, e.g. (0, 255, 0) or [0, 255, 0]
            label: Mark text label
        """
        if not self.is_recording:
            logger.warning("Recording not started, cannot mark frame")
            return

        with self.marker_lock:
            self.pending_marker = (color, label)
            logger.debug(f"Set frame marker: {label}, color={color}")

    def start(self):
        """Start recording"""
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return

        # Reset statistics
        self.first_frame_time = None
        self.last_frame_time = None
        self.frame_count = 0
        self.pending_marker = None

        self.is_recording = True
        self.thread = threading.Thread(target=self._record_wrapper, daemon=True)
        self.thread.start()
        logger.info("Video recording thread started")

    def _record_wrapper(self):
        """Recording wrapper function, supports automatic frame rate downgrade"""
        # Try using configured FPS (e.g. 60)
        success = self._record(self.fps)

        # If failed and configured FPS > 30, try downgrading to 30fps
        if not success and self.fps > 30:
            logger.warning(f"Recording at {self.fps}fps failed, trying to downgrade to 30fps...")
            self.is_recording = True  # Reset flag
            self._record(30)

    def _build_input_cmd(self, fps: int) -> list:
        """Build ffmpeg input command (read raw frames from camera to pipe)"""
        system = platform.system()

        if system == 'Darwin':  # macOS
            # Use fps filter to force limit output frame rate
            return [
                'ffmpeg',
                '-f', 'avfoundation',
                '-framerate', str(fps),
                '-video_size', f'{self.width}x{self.height}',
                '-pixel_format', self.pixel_format,
                '-i', self.camera_name,
                '-vf', f'fps={fps}',  # Force frame rate
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'
            ]
        elif system == 'Linux':
            return [
                'ffmpeg',
                '-f', 'v4l2',
                '-framerate', str(fps),
                '-video_size', f'{self.width}x{self.height}',
                '-pixel_format', self.pixel_format,
                '-i', self.camera_name,
                '-vf', f'fps={fps}',  # Force frame rate
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'
            ]
        elif system == 'Windows':
            return [
                'ffmpeg',
                '-f', 'dshow',
                '-framerate', str(fps),
                '-video_size', f'{self.width}x{self.height}',
                # '-pixel_format', self.pixel_format,
                '-i', f'video={self.camera_name}',
                '-vf', f'fps={fps}',  # Force frame rate
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'
            ]
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

    def _build_output_cmd(self, fps: int, output_file: Path) -> list:
        """Build ffmpeg output command (read frames from pipe and encode to output)"""
        return [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(fps),
            '-i', '-',
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-vsync', 'cfr',
            '-y',
            str(output_file)
        ]

    def _get_bgr_color(self, color: Union[str, tuple, list]) -> tuple:
        """Parse color to BGR tuple"""
        if isinstance(color, str):
            bgr = self.color_map.get(color.lower())
            if bgr is None:
                logger.warning(f"Color '{color}' not defined in layout, using white")
                return (255, 255, 255)
            return tuple(bgr)
        elif isinstance(color, (tuple, list)) and len(color) >= 3:
            return tuple(color[:3])
        else:
            return (255, 255, 255)

    def _draw_marker(self, frame: np.ndarray, color: Union[str, tuple, list], label: str):
        """Draw marker on frame

        Args:
            frame: numpy array, BGR format
            color: Color name or BGR tuple/list
            label: Text label
        """
        bgr_color = self._get_bgr_color(color)

        # Get marker region
        x = self.marker_region.get('x', 10)
        y = self.marker_region.get('y', 10)
        w = self.marker_region.get('width', 100)
        h = self.marker_region.get('height', 100)

        # Draw color box
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, -1)

        # Draw label text (place to the right of color box)
        if label:
            text_x = x + w + 10
            text_y = y + h // 2 + 10
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _record(self, fps: int) -> bool:
        """Recording thread execution function, uses pipe mode for real-time frame marking"""
        success = False

        try:
            # Generate output filename
            if self.timestamp:
                timestamp = self.timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # If retrying (different fps), mark in filename
            filename_suffix = ""
            if fps != self.fps:
                filename_suffix = f"_{fps}fps"

            self.output_file = self.output_dir / f"{self.filename_prefix}_{timestamp}{filename_suffix}.mp4"

            logger.info(f"Recording using ffmpeg pipe mode: {self.camera_name}")
            logger.info(f"Parameters: {self.width}x{self.height} @ {fps}fps")
            logger.info(f"Output: {self.output_file}")

            # Calculate frame size (BGR24: 3 bytes per pixel)
            frame_size = self.width * self.height * 3

            # Start input process (camera -> pipe)
            input_cmd = self._build_input_cmd(fps)
            logger.debug(f"Input command: {' '.join(input_cmd)}")

            self.input_process = subprocess.Popen(
                input_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * 2
            )

            # Start output process (pipe -> file)
            output_cmd = self._build_output_cmd(fps, self.output_file)
            logger.debug(f"Output command: {' '.join(output_cmd)}")

            self.output_process = subprocess.Popen(
                output_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * 2
            )

            logger.info(f"FFmpeg pipe processes started [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

            # Main loop: process frame by frame
            while self.is_recording:
                # Check if input process is still running
                if self.input_process.poll() is not None:
                    # Process ended, check if error
                    if self.input_process.returncode != 0:
                        stderr = self.input_process.stderr.read().decode('utf-8', errors='ignore')
                        logger.error(f"Input process exited abnormally:\n{stderr}")
                    break

                # Read one frame of raw data
                raw_frame = self.input_process.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) < frame_size:
                    logger.debug("Could not read complete frame data")
                    break

                # Record first frame time
                current_time = datetime.now()
                if self.first_frame_time is None:
                    self.first_frame_time = current_time
                    logger.info(f"First frame recorded: {self.first_frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    success = True

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                # Need to copy because frombuffer returns read-only view
                frame = frame.copy()

                # Check and apply marker (consistent with capture.py)
                with self.marker_lock:
                    if self.pending_marker is not None:
                        color, label = self.pending_marker
                        self._draw_marker(frame, color, label)
                        logger.info(f"Applied frame marker: {label} at frame {self.frame_count}")
                        self.pending_marker = None

                # Write to output process
                try:
                    self.output_process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.error("Output pipe closed")
                    break

                self.frame_count += 1
                self.last_frame_time = current_time

            # Recording ended, output statistics
            logger.info("Recording complete")
            if self.last_frame_time:
                logger.info(f"Last frame time: {self.last_frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            logger.info(f"Total frames: {self.frame_count}")

            # Calculate recording duration
            if self.first_frame_time and self.last_frame_time:
                duration = (self.last_frame_time - self.first_frame_time).total_seconds()
                actual_fps = self.frame_count / duration if duration > 0 else 0
                logger.info(f"Recording duration: {duration:.2f}s, actual frame rate: {actual_fps:.2f}fps")

        except Exception as e:
            logger.error(f"Error during recording: {e}", exc_info=True)
        finally:
            # Cleanup processes
            self._cleanup_processes()
            logger.info("Recording resources cleaned up")
            return success

    def _cleanup_processes(self):
        """Cleanup ffmpeg processes"""
        # Close output process's input pipe
        if self.output_process:
            try:
                if self.output_process.stdin:
                    self.output_process.stdin.close()
                self.output_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Output process not responding, force killing")
                self.output_process.kill()
            except Exception as e:
                logger.warning(f"Error closing output process: {e}")
            self.output_process = None

        # Close input process
        if self.input_process:
            try:
                self.input_process.terminate()
                self.input_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Input process not responding, force killing")
                self.input_process.kill()
            except Exception as e:
                logger.warning(f"Error closing input process: {e}")
            self.input_process = None

    def stop(self):
        """Stop recording"""
        if not self.is_recording:
            return

        logger.info("Stopping recording...")
        self.is_recording = False

        # Wait for thread to end (thread will cleanup processes)
        if self.thread:
            self.thread.join(timeout=10)

        logger.info("Video recording stopped")

    def get_output_file(self) -> Optional[str]:
        """Get current recording output file path"""
        return str(self.output_file) if self.output_file else None

    def get_recording_stats(self) -> dict:
        """Get recording statistics"""
        return {
            'first_frame_time': self.first_frame_time,
            'last_frame_time': self.last_frame_time,
            'frame_count': self.frame_count,
            'output_file': self.get_output_file()
        }

    def get_current_frame_number(self) -> int:
        """Get current frame number (useful for debugging mark timing)"""
        return self.frame_count


def create_capture_from_config(
    config_path: str = "configs/default.yaml",
    layout_path: Optional[str] = None,
    filename_prefix: str = "recording",
    timestamp: Optional[str] = None,
    **overrides
) -> Capture:
    """Factory function: create Capture instance from config file

    Args:
        config_path: Config file path
        layout_path: Layout file path (optional, for loading color config)
        filename_prefix: Filename prefix
        timestamp: Timestamp string
        **overrides: Override parameters

    Returns:
        Capture instance
    """
    return Capture(
        config_path=config_path,
        layout_path=layout_path,
        filename_prefix=filename_prefix,
        timestamp=timestamp,
        **overrides
    )
