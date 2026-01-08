#!/usr/bin/env python3
"""
Video Capture Module
Provides high frame rate video capture and recording functionality, supports running in independent threads.
Uses ffmpeg for high-performance video recording.
"""

import logging
import threading
import subprocess
import platform
import ffmpeg
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

camera1080p = "Cisco Desk Camera 1080p"
camera4K = "Cisco Desk Camera 4K"

class Capture:
    """Video capture class for recording high frame rate video using ffmpeg"""
    def __init__(self, camera_name=camera4K, width=1920, height=1080, fps=60, output_dir='recording', pixel_format='uyvy422', filename_prefix='recording', timestamp=None):
        """
        Initialize video recorder.

        Args:
            camera_name: Camera name, default is "Cisco Desk Camera 4K"
            width: Video width
            height: Video height
            fps: Target frame rate, default 60fps
            output_dir: Video output directory
            pixel_format: Pixel format, default uyvy422
            filename_prefix: Video filename prefix
            timestamp: Optional timestamp string
        """
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.pixel_format = pixel_format
        self.filename_prefix = filename_prefix
        self.timestamp = timestamp
        self.is_recording = False
        self.thread = None
        self.ffmpeg_process = None
        self.output_file = None

        # Recording statistics
        self.first_frame_time = None
        self.last_frame_time = None
        self.frame_count = 0

        # Frame marking info
        self.frame_marks = []  # List of {time, color, label}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def mark_frame(self, color, label=""):
        """Mark current frame

        Args:
            color: Mark color (e.g.: 'red', 'green', 'blue', 'yellow')
            label: Mark text label
        """
        if not self.is_recording:
            logger.warning("Recording not started, cannot mark frame")
            return

        mark_time = datetime.now()
        self.frame_marks.append({
            'time': mark_time,
            'color': color,
            'label': label
        })
        logger.info(f"Mark frame: {label} [{color}] at {mark_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    def start(self):
        """Start recording"""
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return

        # Reset statistics
        self.first_frame_time = None
        self.last_frame_time = None
        self.frame_count = 0
        self.frame_marks = []

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
            self.is_recording = True # Reset flag
            self._record(30)

    def _record(self, fps):
        """Recording thread execution function, directly uses ffmpeg for recording"""
        ffmpeg_process = None
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

            logger.info(f"Recording using ffmpeg mode: {self.camera_name}")
            logger.info(f"Parameters: {self.width}x{self.height} @ {fps}fps")
            logger.info(f"Output: {self.output_file}")

            # Build ffmpeg recording command (record directly from camera to file)
            # Select different input format based on operating system
            system = platform.system()
            if system == 'Darwin':  # macOS
                # avfoundation input format
                ffmpeg_process = (
                    ffmpeg
                    .input(self.camera_name,
                           f='avfoundation',
                           framerate=fps,
                           s=f'{self.width}x{self.height}',
                           pix_fmt=self.pixel_format)
                    .output(str(self.output_file),
                            vcodec='libx264',
                            preset='ultrafast',
                            crf=18,
                            pix_fmt='yuv420p',
                            r=fps,  # Force output frame rate
                            vsync='cfr')  # Constant frame rate mode
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stderr=True)
                )
            elif system == 'Linux':
                # Linux uses v4l2 input format
                # camera_name should be device path, e.g. '/dev/video0'
                ffmpeg_process = (
                    ffmpeg
                    .input(self.camera_name,
                           f='v4l2',
                           framerate=fps,
                           s=f'{self.width}x{self.height}',
                           pix_fmt=self.pixel_format)
                    .output(str(self.output_file),
                            vcodec='libx264',
                            preset='ultrafast',
                            crf=18,
                            pix_fmt='yuv420p',
                            r=fps,  # Force output frame rate
                            vsync='cfr')  # Constant frame rate mode
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stderr=True)
                )
            elif system == 'Windows':
                # Windows uses dshow (DirectShow) input format
                # camera_name should be device name, e.g. 'Integrated Camera'
                # Note: dshow device name format is "video=device_name"
                ffmpeg_cmd = (
                    ffmpeg
                    .input(f'video={self.camera_name}',
                           f='dshow',
                           framerate=fps,
                           s=f'{self.width}x{self.height}',
                           # pix_fmt=self.pixel_format
                           )
                    .output(str(self.output_file),
                            vcodec='libx264',
                            preset='ultrafast',
                            crf=18,
                            pix_fmt='yuv420p',
                            r=fps,  # Force output frame rate
                            vsync='cfr')  # Constant frame rate mode
                    .overwrite_output()
                )
                # Print the ffmpeg command
                cmd_args = ffmpeg_cmd.compile()
                logger.info(f"FFmpeg command: {' '.join(cmd_args)}")
                ffmpeg_process = ffmpeg_cmd.run_async(pipe_stdin=True, pipe_stderr=True)
            else:
                raise RuntimeError(f"Unsupported operating system: {system}")

            logger.info(f"FFmpeg recording process started, [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
            success = True

            # Save process reference for stop to use
            self.ffmpeg_process = ffmpeg_process

            # Real-time read ffmpeg output to get frame info
            import re
            frame_pattern = re.compile(r'frame=\s*(\d+)')

            # ffmpeg uses \r to update the same line, need to read char by char
            buffer = b''
            while self.is_recording:
                # Check if process is still running
                if ffmpeg_process.poll() is not None:
                    break

                try:
                    # Read one byte
                    char = ffmpeg_process.stderr.read(1)
                    if not char:
                        break

                    buffer += char

                    # Process buffer when encountering carriage return or newline
                    if char in (b'\r', b'\n'):
                        if buffer:
                            line_str = buffer.decode('utf-8', errors='ignore')

                            # Parse frame count
                            match = frame_pattern.search(line_str)
                            if match:
                                current_frame = int(match.group(1))
                                current_time = datetime.now()

                                # Record first frame time
                                if self.first_frame_time is None and current_frame > 0:
                                    # Subtract estimated processing delay (~2 frame time) to improve accuracy
                                    # This compensates for ffmpeg output delay and Python parsing delay
                                    estimated_delay = 2 / fps
                                    self.first_frame_time = current_time - timedelta(seconds=estimated_delay)
                                    logger.info(f"First frame recorded: {self.first_frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (corrected {estimated_delay*1000:.1f}ms delay)")

                                # Update frame count and last frame time
                                self.frame_count = current_frame
                                self.last_frame_time = current_time

                            buffer = b''

                except Exception as e:
                    logger.debug(f"Error parsing ffmpeg output: {e}")
                    break

            # Wait for process to complete
            ffmpeg_process.wait()

            logger.info("Recording complete")
            if self.last_frame_time:
                logger.info(f"Last frame time: {self.last_frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            logger.info(f"Total frames: {self.frame_count}")

            # Calculate recording duration
            actual_avg_fps = self.fps  # Default to configured FPS
            if self.first_frame_time and self.last_frame_time:
                duration = (self.last_frame_time - self.first_frame_time).total_seconds()
                fps = self.frame_count / duration if duration > 0 else 0
                logger.info(f"Recording duration: {duration:.2f}s, calculated frame rate: {fps:.2f}fps")
                actual_avg_fps = fps  # Update with calculated FPS

                # Use ffprobe to get actual video frame rate
                try:
                    probe_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-select_streams', 'v:0',
                        '-show_entries', 'stream=avg_frame_rate,r_frame_rate',
                        '-of', 'json',
                        str(self.output_file)
                    ]
                    probe_result = subprocess.run(
                        probe_cmd,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if probe_result.returncode == 0:
                        import json
                        probe_data = json.loads(probe_result.stdout)
                        if 'streams' in probe_data and len(probe_data['streams']) > 0:
                            stream = probe_data['streams'][0]
                            # Display two frame rates: avg_frame_rate (actual average) and r_frame_rate (declared)
                            if 'avg_frame_rate' in stream:
                                fps_str = stream['avg_frame_rate']
                                num, den = map(int, fps_str.split('/'))
                                actual_avg_fps = num / den
                                logger.info(f"Video average frame rate (ffprobe avg_frame_rate): {actual_avg_fps:.2f}fps")
                            if 'r_frame_rate' in stream:
                                fps_str = stream['r_frame_rate']
                                num, den = map(int, fps_str.split('/'))
                                r_fps = num / den
                                logger.info(f"Video declared frame rate (ffprobe r_frame_rate): {r_fps:.2f}fps")
                except subprocess.TimeoutExpired:
                    logger.warning("ffprobe query timed out")
                except Exception as e:
                    logger.debug(f"Error getting video frame rate: {e}")

            # Process frame marks
            if self.frame_marks and self.first_frame_time:
                self._apply_marks(actual_avg_fps)

        except ffmpeg.Error as e:
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg recording error:\n{stderr}")
        except Exception as e:
            logger.error(f"Error during recording: {e}", exc_info=True)
        finally:
            # Cleanup process
            if ffmpeg_process:
                try:
                    if ffmpeg_process.poll() is None:
                        ffmpeg_process.terminate()
                        try:
                            ffmpeg_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning("Process not responding, force killing")
                            ffmpeg_process.kill()
                except Exception as e:
                    logger.warning(f"Error closing process: {e}")
                    try:
                        ffmpeg_process.kill()
                    except:
                        pass

            self.ffmpeg_process = None
            logger.info("Recording resources cleaned up")
            return success

    def _apply_marks(self, fps):
        """Apply marks to video, output new video with marks

        Args:
            fps: Video's actual average frame rate
        """
        if not self.output_file or not self.output_file.exists():
            logger.warning("Output file does not exist, cannot apply marks")
            return

        logger.info(f"Processing {len(self.frame_marks)} frame marks...")
        logger.info(f"Using frame rate {fps:.2f}fps to calculate mark frame positions")

        # Create mapping of frame number to label
        mark_map = {}
        for mark in self.frame_marks:
            # Calculate time offset from first frame to mark time
            time_offset = (mark['time'] - self.first_frame_time).total_seconds()

            # Use actual average frame rate to calculate corresponding frame number (0-indexed)
            frame_number = int(time_offset * fps)

            # Ensure frame number is within valid range
            if 0 <= frame_number < self.frame_count:
                # If same frame has multiple marks, combine labels
                if frame_number in mark_map:
                    mark_map[frame_number]['labels'].append(mark['label'])
                    mark_map[frame_number]['colors'].append(mark['color'])
                else:
                    mark_map[frame_number] = {
                        'labels': [mark['label']],
                        'colors': [mark['color']]
                    }
                logger.info(f"Mark '{mark['label']}' mapped to frame {frame_number} (time offset: {time_offset:.3f}s)")
            else:
                logger.warning(f"Mark '{mark['label']}' out of valid range (frame: {frame_number}, total: {self.frame_count})")

        if not mark_map:
            logger.info("No valid marks to apply")
            return

        logger.info(f"Mark frame mapping: {len(mark_map)} marked frames")

        # Color mapping: convert color names to hex color codes (from jupiter.yaml)
        color_map = {
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'magenta': 'FF00FF',
            'orange': 'FFA500',
            'purple': '800080',
            'pink': 'FFC0CB',
            'white': 'FFFFFF',
            'lime': '80FF00',
            'turquoise': '40E0D0',
            'navy': '000080',
            'maroon': '800000',
            'olive': '808000',
            'teal': '008080',
            'coral': 'FF7F50'
        }

        try:
            # Build ffmpeg filter chain
            filter_parts = []

            for frame_num, mark_info in sorted(mark_map.items()):
                # Add draw filter for each marked frame
                for i, (label, color) in enumerate(zip(mark_info['labels'], mark_info['colors'])):
                    # Get color code
                    color_code = color_map.get(color.lower(), 'FFFFFF')

                    # Mark region position (top-left corner)
                    box_x = 10
                    box_y = 10 + i * 110  # Each mark 110 pixels apart vertically
                    box_width = 100
                    box_height = 100

                    # Add color box (only show on specific frame)
                    drawbox = (
                        f"drawbox=x={box_x}:y={box_y}:w={box_width}:h={box_height}:"
                        f"color=0x{color_code}:thickness=fill:"
                        f"enable='eq(n\\,{frame_num})'"
                    )
                    filter_parts.append(drawbox)
                    logger.info(f"Added mark filter: Frame {frame_num} - {label} ({color})")

            # Combine all filters
            vf = ','.join(filter_parts)

            # Generate temporary output filename
            temp_output = self.output_file.parent / f"{self.output_file.stem}_marked_temp.mp4"

            logger.info(f"Generating marked video...")
            logger.info(f"Applying {len(filter_parts)} filters to video...")

            # Use subprocess to call ffmpeg directly (Avoid ffmpeg-python quote handling issues)
            cmd = [
                'ffmpeg',
                '-i', str(self.output_file),
                '-vf', vf,
                '-vcodec', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite output file
                str(temp_output)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg processing failed:\n{result.stderr}")
            else:
                # Delete original file and rename marked file
                self.output_file.unlink()
                temp_output.rename(self.output_file)
                logger.info(f"Marked video generated: {self.output_file}")
                logger.info(f"Processed {len(mark_map)} marked frames")

        except ffmpeg.Error as e:
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"Error processing video marks:\n{stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg call failed: {e}")
        except Exception as e:
            logger.error(f"Error processing marks: {e}", exc_info=True)

    def stop(self):
        """Stop recording."""
        if not self.is_recording:
            return

        logger.info("Stopping recording...")
        self.is_recording = False

        # Send 'q' to ffmpeg for graceful shutdown (allows it to finalize the file)
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(b'q')
                self.ffmpeg_process.stdin.flush()
            except:
                # If that fails, terminate forcefully
                try:
                    self.ffmpeg_process.terminate()
                except:
                    pass

        # Wait for thread to end
        if self.thread:
            self.thread.join(timeout=10)

        logger.info("Video recording stopped")

    def get_output_file(self):
        """Get current recording output file path."""
        return str(self.output_file) if self.output_file else None

    def get_recording_stats(self):
        """Get recording statistics."""
        return {
            'first_frame_time': self.first_frame_time,
            'last_frame_time': self.last_frame_time,
            'frame_count': self.frame_count,
            'output_file': self.get_output_file()
        }
