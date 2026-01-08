#!/usr/bin/env python3
"""
Observer Program
Analyzes video to detect key events based on color markers and compares them with action sequences in sequences/*.yaml.
"""

import cv2
import yaml
import os
import argparse
import numpy as np
import logging
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from robotarm.vlm import create_vlm_client, VLMClient
from robotarm.logging_utils import setup_logging
from robotarm.utils import get_resource_path

logger = logging.getLogger(__name__)


@dataclass
class KeyEvent:
    """Key Event"""
    frame_index: int
    key_name: str
    color_name: str
    timestamp_ms: float


def parse_vlm_response(response: str) -> bool:
    """Parse JSON response from VLM"""
    if not response:
        return False
    try:
        # Try to parse JSON directly
        data = json.loads(response)
        return data.get("result", False)
    except json.JSONDecodeError:
        # Try to find JSON pattern
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return data.get("result", False)
            except:
                pass
    return False


def print_vlm_stats(results: list):
    """Print VLM UI response time statistics"""
    if not results:
        return

    logger.info(f"\n{'='*90}")
    logger.info(f"UI Response Time Statistics")
    logger.info(f"{'='*90}")
    logger.info(f"{'Event':<15} | {'Response (ms)':<15} | {'Frame':<8} | {'Status':<10} | {'Prompt'}")
    logger.info("-" * 90)

    for res in results:
        status = "✅ Detected" if res['found'] else "❌ Not Detected"
        time_str = f"{res['response_time_ms']:.1f}" if res['response_time_ms'] is not None else "-"
        frame_str = str(res.get('found_frame', '-'))
        # Truncate long prompt
        prompt = res['prompt']
        if len(prompt) > 30:
            prompt = prompt[:27] + "..."
        logger.info(f"{res['event']:<15} | {time_str:<15} | {frame_str:<8} | {status:<10} | {prompt}")
    logger.info("-" * 90)


def load_yaml(path: str) -> Optional[dict]:
    """Load YAML file"""
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse file {path}: {e}")
            return None


def build_color_to_key_map(layout_config: dict) -> dict:
    """
    Build Color Name -> Key Name map from layout config
    """
    color_to_key = {}
    buttons = layout_config.get('device', {}).get('buttons', [])

    for button in buttons:
        capture = button.get('capture', {})
        color_name = capture.get('color')
        if color_name:
            color_to_key[color_name] = button['name']

    return color_to_key


def build_color_bgr_map(layout_config: dict) -> dict:
    """
    Build Color Name -> BGR Value map from layout config
    """
    return layout_config.get('device', {}).get('colors', {})


def get_rotation_code(angle: int):
    """Get OpenCV rotation code"""
    if angle == 90:
        return cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        return cv2.ROTATE_180
    elif angle == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def color_distance(c1, c2) -> float:
    """Calculate Euclidean distance between two colors"""
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))


def order_points(pts):
    """Order four points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_screen_trapezoid(frame, min_area: int = 10000):
    """
    Detect light screen area (trapezoid) in the frame
    Returns: Four corner coordinates (TL, TR, BR, BL) or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < min_area:
        return None

    hull = cv2.convexHull(largest_contour)

    for eps_factor in [0.02, 0.03, 0.04, 0.05]:
        epsilon = eps_factor * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return order_points(pts)

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    pts = box.astype(np.float32)
    return order_points(pts)


def perspective_transform(frame, src_points, output_size):
    """Correct trapezoid to rectangle using perspective transform"""
    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, output_size)
    return warped


def get_reference_frame(cap, ref_index: int):
    """Get reference frame"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ref_index < 0:
        ref_index = total_frames + ref_index

    ref_index = max(0, min(ref_index, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_index)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if ret:
        return frame, ref_index
    return None, ref_index


def detect_marker_color(
    frame,
    config: dict,
    color_bgr_map: dict,
    debug: bool = False,
    frame_index: int = 0
) -> tuple[Optional[str], float, tuple]:
    """
    Detect color marker in the frame
    Returns: (Matched color name or None, Min distance, Average BGR)
    """
    vp_config = config.get('video_processing', {})
    capture_config = config.get('capture', {})
    # Prefer marker_region from capture, fallback to video_processing (backward compatibility)
    region = capture_config.get('marker_region') or vp_config.get('marker_region', {'x': 10, 'y': 10, 'width': 100, 'height': 100})
    threshold = vp_config.get('color_threshold', 50)

    x, y, w, h = region['x'], region['y'], region['width'], region['height']

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return None, float('inf'), (0, 0, 0)

    roi = frame[y:y+h, x:x+w]
    avg_color = np.mean(roi, axis=(0, 1))

    min_dist = float('inf')
    matched_color = None

    for color_name, bgr in color_bgr_map.items():
        dist = color_distance(avg_color, bgr)
        if dist < min_dist:
            min_dist = dist
            matched_color = color_name

    if debug and frame_index % 30 == 0:  # Output once per second
        logger.debug(
            f"Frame {frame_index}: Region BGR={avg_color.astype(int).tolist()}, "
            f"Nearest match: {matched_color} (Dist={min_dist:.1f}, Threshold={threshold})"
        )

    if min_dist < threshold:
        return matched_color, min_dist, tuple(avg_color.astype(int))

    return None, min_dist, tuple(avg_color.astype(int))


def observe_video(
    video_path: str,
    config: dict,
    layout_config: dict,
    melody: Optional[dict] = None,
    vlm_client: Optional[VLMClient] = None,
    save_frames: bool = False,
    save_corrected: bool = False,
    debug: bool = False
) -> list[KeyEvent]:
    """
    Analyze video and detect key events
    """
    video_name = Path(video_path).stem
    frames_dir = Path('frames') / video_name
    images_dir = Path('images') / video_name

    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw frames output directory: {frames_dir}")

    if save_corrected:
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Corrected frames output directory: {images_dir}")

    logger.info(f"Start analyzing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")

    vp_config = config.get('video_processing', {})
    rotate_angle = vp_config.get('rotate', 0)
    rotation_code = get_rotation_code(rotate_angle)

    # Screen correction config
    screen_config = vp_config.get('screen_correction', {})
    ref_frame_index = screen_config.get('reference_frame', -1)
    output_ratio = screen_config.get('output_ratio', [3, 4])
    output_height = screen_config.get('output_height', 640)
    output_width = int(output_height * output_ratio[0] / output_ratio[1])
    output_size = (output_width, output_height)

    # Build color maps
    color_bgr_map = build_color_bgr_map(layout_config)
    color_to_key = build_color_to_key_map(layout_config)

    # Debug mode: Enable DEBUG logging
    if debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        vp_config = config.get('video_processing', {})
        capture_config = config.get('capture', {})
        region = capture_config.get('marker_region') or vp_config.get('marker_region', {})
        threshold = vp_config.get('color_threshold', 50)
        logger.debug(f"Marker region: x={region.get('x')}, y={region.get('y')}, "
                    f"w={region.get('width')}, h={region.get('height')}")
        logger.debug(f"Color threshold: {threshold}")
        logger.debug(f"Rotation angle: {rotate_angle}")

    # Screen detection mode:
    # reference_frame = -1: Detect once using last frame
    # reference_frame = 0: Detect per frame
    # reference_frame > 0: Detect once using specified frame
    screen_pts = None
    per_frame_detection = (ref_frame_index == 0)
    need_screen_detection = save_corrected or vlm_client

    if need_screen_detection and not per_frame_detection:
        # Detect using fixed reference frame
        ref_frame, actual_ref_index = get_reference_frame(cap, ref_frame_index)
        if ref_frame is not None:
            if rotation_code is not None:
                ref_frame = cv2.rotate(ref_frame, rotation_code)
            screen_pts = detect_screen_trapezoid(ref_frame)
            if screen_pts is not None:
                logger.info(f"Detected screen area from reference frame {actual_ref_index}")
            else:
                if save_corrected:
                    logger.warning("Cannot detect screen area, skipping corrected frame saving")
                    save_corrected = False
                if vlm_client:
                    logger.warning("Cannot detect screen area, VLM will use raw frames")
    elif need_screen_detection and per_frame_detection:
        logger.info("Enabled per-frame screen detection mode")

    events: list[KeyEvent] = []
    prev_color = None
    frame_count = 0

    # VLM State Tracking
    melody_actions = [a for a in melody.get('actions', []) if a.get('type') == 'button'] if melody else []
    melody_idx = 0
    active_vlm_task = None  # { 'event_name': str, 'prompt': str, 'start_frame': int, 'start_time': float, 'frames': [] }
    vlm_results = [] # [{'event': str, 'prompt': str, 'response_time_ms': float, 'found': bool}]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Rotate frame (for screen detection and correction)
            if rotation_code is not None:
                rotated_frame = cv2.rotate(frame, rotation_code)
            else:
                rotated_frame = frame

            # Detect screen per frame (if enabled)
            if per_frame_detection and need_screen_detection:
                screen_pts = detect_screen_trapezoid(rotated_frame)

            # Prepare frame for VLM (prefer corrected frame)
            vlm_frame = frame
            corrected_frame = None
            if screen_pts is not None:
                corrected_frame = perspective_transform(rotated_frame, screen_pts, output_size)
                vlm_frame = corrected_frame

            # Detect color marker (on raw frame)
            detected_color, min_dist, avg_color = detect_marker_color(
                frame, config, color_bgr_map, debug=debug, frame_index=frame_count
            )

            # Debug mode: Save first frame with marker region
            if debug and frame_count == 0:
                debug_frame = frame.copy()
                region = config.get('capture', {}).get('marker_region') or config.get('video_processing', {}).get('marker_region', {})
                x, y = region.get('x', 0), region.get('y', 0)
                w, h = region.get('width', 100), region.get('height', 100)
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"BGR: {avg_color}", (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                debug_path = Path('debug_marker_region.jpg')
                cv2.imwrite(str(debug_path), debug_frame)
                logger.info(f"Debug image saved: {debug_path}")

            current_event_name = None
            # Detect new color change (from nothing to something, or color change)
            if detected_color is not None and detected_color != prev_color:
                key_name = color_to_key.get(detected_color, f"Unknown({detected_color})")
                current_event_name = key_name
                event = KeyEvent(
                    frame_index=frame_count,
                    key_name=key_name,
                    color_name=detected_color,
                    timestamp_ms=timestamp_ms
                )
                events.append(event)
                logger.info(
                    f"Frame {frame_count:6d} ({timestamp_ms/1000:.2f}s): "
                    f"Detected {key_name} ({detected_color}) [Dist={min_dist:.1f}]"
                )

                # --- VLM Logic Start ---

                # 1. If there is an active VLM task, it means no UI response detected before next event
                if active_vlm_task and vlm_client:
                    logger.warning("❌ UI response not detected (reached next event)")
                    vlm_results.append({
                        'event': active_vlm_task['event_name'],
                        'prompt': active_vlm_task['prompt'],
                        'response_time_ms': None,
                        'found': False,
                        'found_frame': None
                    })
                    active_vlm_task = None

                # 2. Check if current key matches expected action in Melody, and check for 'ask'
                if melody_idx < len(melody_actions):
                    expected_action = melody_actions[melody_idx]
                    if expected_action['name'] == key_name:
                        # Match success
                        if 'ask' in expected_action:
                            ask_prompt = expected_action['ask']
                            logger.info(f"Start VLM task: '{ask_prompt}'")
                            active_vlm_task = {
                                'event_name': key_name,
                                'prompt': ask_prompt,
                                'start_frame': frame_count,
                                'start_time': timestamp_ms
                            }
                        melody_idx += 1
                    else:
                        # Mismatch, could be false detection or miss, skip simply here, keep melody_idx unchanged
                        pass

                # --- VLM Logic End ---

            prev_color = detected_color

            # Per-frame VLM detection
            if active_vlm_task and vlm_client:
                resp = vlm_client.ask(vlm_frame, active_vlm_task['prompt'])
                if parse_vlm_response(resp):
                    # Calculate response time
                    frame_delta = frame_count - active_vlm_task['start_frame']
                    time_delta_ms = frame_delta * (1000.0 / fps) if fps > 0 else 0

                    logger.info(f"✅ UI response detected! Time: {time_delta_ms:.1f} ms (Frame delta: {frame_delta}) @ Frame {frame_count}")

                    vlm_results.append({
                        'event': active_vlm_task['event_name'],
                        'prompt': active_vlm_task['prompt'],
                        'response_time_ms': time_delta_ms,
                        'found': True,
                        'found_frame': frame_count
                    })
                    active_vlm_task = None

            # Save frames (raw)
            if save_frames:
                filename = f"frame_{frame_count:06d}"
                if current_event_name:
                    filename += f"_{current_event_name}"
                frame_path = frames_dir / f"{filename}.jpg"
                cv2.imwrite(str(frame_path), frame)

            if save_corrected and corrected_frame is not None:
                filename = f"frame_{frame_count:06d}"
                if current_event_name:
                    filename += f"_{current_event_name}"
                corrected_path = images_dir / f"{filename}.jpg"
                cv2.imwrite(str(corrected_path), corrected_frame)

            frame_count += 1

            if frame_count % 500 == 0:
                print(f"Analyzed {frame_count}/{total_frames} frames...", end='\r')

        # Video ended, if VLM task still active, means not detected
        if active_vlm_task and vlm_client:
            logger.warning("❌ UI response not detected (Video ended)")
            vlm_results.append({
                'event': active_vlm_task['event_name'],
                'prompt': active_vlm_task['prompt'],
                'response_time_ms': None,
                'found': False,
                'found_frame': None
            })

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    finally:
        cap.release()
        print()

    # Print VLM stats
    if vlm_results:
        print_vlm_stats(vlm_results)

    logger.info(f"Analysis complete. Total frames: {frame_count}, Detected {len(events)} key events")
    return events


def compare_with_melody(events: list[KeyEvent], melody: dict) -> None:
    """
    Compare detected key events with action sequence in melody
    """
    actions = melody.get('actions', [])
    button_actions = [a for a in actions if a.get('type') == 'button']

    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing melody sequence: {melody.get('name', 'Unnamed')}")
    logger.info(f"{'='*60}")

    logger.info(f"\nKey actions in Melody ({len(button_actions)}):")
    for i, action in enumerate(button_actions):
        logger.info(f"  {i+1}. {action.get('name')} - {action.get('comment', '')}")

    logger.info(f"\nDetected key events ({len(events)}):")
    for i, event in enumerate(events):
        logger.info(
            f"  {i+1}. {event.key_name} @ Frame {event.frame_index} "
            f"({event.timestamp_ms/1000:.2f}s)"
        )

    # Simple comparison
    logger.info(f"\nComparison Result:")
    min_len = min(len(button_actions), len(events))

    matches = 0
    for i in range(min_len):
        expected = button_actions[i].get('name')
        actual = events[i].key_name
        status = "✓" if expected == actual else "✗"
        if expected == actual:
            matches += 1
        logger.info(f"  {i+1}. Expect: {expected:15s} Actual: {actual:15s} {status}")

    if len(button_actions) > len(events):
        logger.warning(f"  Detected fewer keys than expected ({len(events)} < {len(button_actions)})")
    elif len(events) > len(button_actions):
        logger.warning(f"  Detected more keys than expected ({len(events)} > {len(button_actions)})")

    logger.info(f"\nMatch Rate: {matches}/{min_len} ({100*matches/min_len:.1f}%)" if min_len > 0 else "\nCannot compare")


def run_observer(args=None):
    setup_logging("Observer")
    # Suppress httpx logging to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='Observe video and detect key events')
    parser.add_argument('video_file', help='Path to video file to analyze')
    parser.add_argument(
        '-m', '--melody',
        default='dial_8890.yaml',
        help='Score config filename or path (Default: dial_8890.yaml)'
    )

    if args is not None:
        parsed_args = parser.parse_args(args)
    else:
        parsed_args = parser.parse_args()

    # Load melody
    try:
        melody_path = get_resource_path(parsed_args.melody, 'scores')
        with open(melody_path, 'r', encoding='utf-8') as f:
            melody = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Melody file not found: {parsed_args.melody}")
        return 1

    if not melody:
        return 1

    # Try to extract info from video filename to set log filename
    video_path = Path(parsed_args.video_file)
    video_stem = video_path.stem  # remove extension

    log_filename = f"{video_stem}_observer.log"
    setup_logging("Observer", filename=log_filename)
    logger.info(f"Log filename set from video: {log_filename}")

    # Load config (refer from melody or use default)
    config_file_name = melody.get('config_file', 'default.yaml')
    try:
        config_path = get_resource_path(config_file_name, 'configs')
        config = load_yaml(str(config_path))
    except FileNotFoundError:
        # Fallback: if filename has path components, try just the filename in resources
        if '/' in config_file_name or '\\' in config_file_name:
            filename_only = Path(config_file_name).name
            try:
                config_path = get_resource_path(filename_only, 'configs')
                logger.info(f"Fallback: Loading built-in config: {config_path}")
                config = load_yaml(str(config_path))
            except FileNotFoundError:
                config = None
        else:
            config = None
        
    if not config:
        logger.error(f"Cannot load config file: {config_file_name}")
        return 1

    # Read observer config from melody
    observer_config = melody.get('observer', {})
    save_frames = observer_config.get('save_frames', False)
    save_corrected = observer_config.get('save_corrected', False)
    do_compare = observer_config.get('compare', True)
    debug = observer_config.get('debug', False)

    # Load layout
    # Prefer device config
    device_name = config.get('device')
    layout_file_name = config.get('layout_file')
    
    layout_path = None
    
    if device_name:
        try:
            layout_path = get_resource_path(f"{device_name}.yaml", 'layouts')
            logger.info(f"Using device layout: {device_name} ({layout_path})")
        except FileNotFoundError:
            logger.warning(f"Layout file for device {device_name} not found")

    # Fallback to layout_file
    if not layout_path and layout_file_name:
        # If it contains directory separators, take just the filename if using resource path logic
        if '/' in layout_file_name or '\\' in layout_file_name:
            layout_file_name = Path(layout_file_name).name
            
        try:
            layout_path = get_resource_path(layout_file_name, 'layouts')
        except FileNotFoundError:
            pass
            
    if not layout_path:
        # Last resort: Default Jupiter
        try:
            layout_path = get_resource_path('jupiter.yaml', 'layouts')
            logger.warning("No layout specified, using default Jupiter layout")
        except FileNotFoundError:
            pass

    if layout_path:
        layout_config = load_yaml(str(layout_path))
    else:
        layout_config = None

    if not layout_config:
        logger.error(f"Cannot load layout file: {layout_file_name}")
        return 1

    # Initialize VLM Client
    vlm_client = create_vlm_client(config)

    # Check video file
    if not os.path.exists(parsed_args.video_file):
        logger.error(f"Video file does not exist: {parsed_args.video_file}")
        return 1

    # Analyze video
    events = observe_video(
        parsed_args.video_file,
        config,
        layout_config,
        melody=melody,
        vlm_client=vlm_client,
        save_frames=save_frames,
        save_corrected=save_corrected,
        debug=debug
    )

    # Compare with melody
    if do_compare:
        compare_with_melody(events, melody)

    return 0


if __name__ == "__main__":
    exit(run_observer())
