#!/usr/bin/env python3
"""
Video Capture Module
Provides high frame rate video capture and recording functionality, supports running in independent threads.
"""

import logging
import threading
import queue
import cv2
import os
from pathlib import Path
from datetime import datetime
from cv2_enumerate_cameras import enumerate_cameras

logger = logging.getLogger(__name__)

camera1080p = "Cisco Desk Camera 1080p"
camera4K = "Cisco Desk Camera 4K"

class Capture:
    """Video capture class for recording high frame rate video."""
    def __init__(self, camera_name=camera4K, width=4096, height=2160, fps=30, output_dir='recording', filename_prefix='recording', timestamp=None):
        """
        Initialize video recorder.

        Args:
            camera_name: Camera name, default is "Cisco Desk Camera 4K"
            fps: Target frame rate, default 60fps (actual frame rate depends on hardware support)
            output_dir: Video output directory
            filename_prefix: Video filename prefix
            timestamp: Optional timestamp string, used for filename generation if provided
        """
        self.camera_id = self._get_camera_id_by_name(camera_name)
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.timestamp = timestamp
        self.is_recording = False
        self.thread = None
        self.cap = None
        self.out = None
        self.marker_lock = threading.Lock()
        self.pending_marker = None  # Store pending marker (color, label)
        self.marker_region = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        self.frame_queue = queue.Queue(maxsize=60)  # Frame buffer queue, approx 2 seconds buffer
        self.write_thread = None

    def set_marker_region(self, region):
        """Set marker region configuration.

        Args:
            region: Dictionary containing x, y, width, height
        """
        if not isinstance(region, dict):
            return
        self.marker_region.update(region)
        logger.info(f"Marker region updated: {self.marker_region}")

    def _get_camera_id_by_name(self, name):
        """Get camera ID by camera name."""
        cameras = enumerate_cameras()
        for cam_id, cam_name in enumerate(cameras):
            if name.lower() in str(cam_name).lower():
                logger.info(f"Found camera '{cam_name}', ID: {cam_id}")
                return cam_id
        logger.error(f"Camera with name containing '{name}' not found, using default camera ID 0")
        return 0

    def mark_frame(self, color, label=""):
        """Draw marker on the next frame.

        Args:
            color: Color tuple in BGR format, e.g., (0, 255, 0) for green
            label: Text label for the marker
        """
        with self.marker_lock:
            self.pending_marker = (color, label)
            logger.debug(f"Set frame marker: {label}, color={color}")

    def _write_worker(self):
        """Write thread function."""
        write_count = 0
        while True:
            frame = self.frame_queue.get()
            if frame is None:  # End signal
                self.frame_queue.task_done()
                break

            if self.out:
                self.out.write(frame)
                write_count += 1
            self.frame_queue.task_done()
        logger.info(f"Write thread ended, total written {write_count} frames")

    def start(self):
        """Start recording."""
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Creating output directory: {self.output_dir}")
            except OSError as e:
                logger.error(f"Cannot create output directory {self.output_dir}: {e}")
                return

        self.is_recording = True
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()
        logger.info("Video recording thread started")

    def _record(self):
        """Recording thread execution function."""
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)

        # Get actual camera parameters and try to set parameters
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Current camera parameters: {width}x{height} @ {actual_fps}fps")
        logger.info(f"Attempting to set camera parameters: {self.width}x{self.height} @ {self.fps}fps")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera parameters after setting: {width}x{height} @ {actual_fps}fps")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer delay

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            self.is_recording = False
            return

        # Create video writer
        if self.timestamp:
            timestamp = self.timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use Path to handle paths, ensuring cross-platform compatibility
        output_path = Path(self.output_dir) / f"{self.filename_prefix}_{timestamp}.mp4"
        output_file = str(output_path)

        # Use actual fps supported by camera, not target fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, actual_fps, (width, height))

        if not self.out.isOpened():
            logger.error(f"Cannot create video file: {output_file}")
            self.cap.release()
            self.is_recording = False
            return

        logger.info(f"Start recording video to: {output_file}")

        # Start write thread
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()

        frame_count = 0

        try:
            while self.is_recording:
                ret, frame = self.cap.read()
                if ret:
                    # Check if there is a pending marker to draw
                    with self.marker_lock:
                        if self.pending_marker is not None:
                            color, label = self.pending_marker

                            # Get region parameters
                            x = self.marker_region.get('x', 0)
                            y = self.marker_region.get('y', 0)
                            w = self.marker_region.get('width', 100)
                            h = self.marker_region.get('height', 100)

                            # Draw color square
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

                            # If there is a label, put text next to the color block (not covering the block)
                            if label:
                                text_x = x + w + 10
                                text_y = y + h // 2 + 10
                                cv2.putText(frame, label, (text_x, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            logger.debug(f"Applied frame marker: {label} at frame {frame_count}")
                            self.pending_marker = None

                    # Put into queue
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    else:
                        logger.warning("Write queue is full, dropping frame")

                    frame_count += 1
                else:
                    logger.warning("Failed to read frame")
                    break
        except Exception as e:
            logger.error(f"Error during recording: {e}")
        finally:
            logger.info(f"Capture ended, captured {frame_count} frames total")
            # Notify write thread to end
            self.frame_queue.put(None)
            if self.write_thread:
                self.write_thread.join()

    def stop(self):
        """Stop recording."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for thread to end
        if self.thread:
            self.thread.join(timeout=5)

        # Release resources
        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()

        logger.info("Video recording stopped")
