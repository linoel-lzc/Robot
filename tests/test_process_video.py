#!/usr/bin/env python3
"""
Video Processing Script
Parses videos in the recording directory, extracts image frames, and detects key markers and rotates images based on configs/default.yaml configuration.
"""

import cv2
import yaml
import os
import glob
import numpy as np
import logging
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robotarm.utils import get_resource_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_yaml(path):
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

def get_markers_from_layout(layout_config):
    """Extract color markers from layout configuration"""
    markers = []
    if not layout_config:
        return markers

    colors = layout_config.get('device', {}).get('colors', {})
    for name, color in colors.items():
        markers.append({
            'name': name.capitalize(),
            'color': color
        })
    return markers

def get_rotation_code(angle):
    """Get OpenCV rotation code"""
    if angle == 90:
        return cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        return cv2.ROTATE_180
    elif angle == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None

def color_distance(c1, c2):
    """Calculate Euclidean distance between two colors"""
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

def order_points(pts):
    """
    Order four points in order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left point has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest x-y difference, bottom-left has largest x-y difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detect_screen_trapezoid(frame, min_area=10000):
    """
    Detect light screen area (trapezoid) in the frame
    Returns: Four corner coordinates (TL, TR, BR, BL) or None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use Otsu threshold to automatically find bright regions
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to fill small holes and smooth edges
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (assumed to be the screen)
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < min_area:
        return None

    # Use convex hull to simplify contour
    hull = cv2.convexHull(largest_contour)

    # Approximate polygon, try to get quadrilateral
    for eps_factor in [0.02, 0.03, 0.04, 0.05]:
        epsilon = eps_factor * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return order_points(pts)

    # If cannot approximate to quadrilateral, use minimum area rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    pts = box.astype(np.float32)
    return order_points(pts)

def perspective_transform(frame, src_points, output_size):
    """
    Perspective transform to correct trapezoid to rectangle
    output_size: (width, height) output dimensions
    """
    dst_points = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, output_size)

    return warped

def get_reference_frame(cap, ref_index):
    """
    Get reference frame
    ref_index: Frame index, -1 means the last frame
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ref_index < 0:
        ref_index = total_frames + ref_index

    ref_index = max(0, min(ref_index, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_index)
    ret, frame = cap.read()

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if ret:
        return frame, ref_index
    return None, ref_index

def detect_marker(frame, config, markers):
    """
    Detect color marker in the frame
    Returns: (detected, marker_name)
    """
    vp_config = config.get('video_processing', {})
    capture_config = config.get('capture', {})
    # marker_region prefers capture config, fallback to video_processing
    region = capture_config.get('marker_region') or vp_config.get('marker_region', {'x': 10, 'y': 10, 'width': 100, 'height': 100})
    threshold = vp_config.get('color_threshold', 50)

    x, y, w, h = region['x'], region['y'], region['width'], region['height']

    # Ensure region is within image bounds
    if y+h > frame.shape[0] or x+w > frame.shape[1]:
        return False, None

    # Extract marker region
    roi = frame[y:y+h, x:x+w]

    # Calculate average color (BGR)
    avg_color = np.mean(roi, axis=(0, 1))

    # Compare with predefined colors
    min_dist = float('inf')
    matched_name = None

    for marker in markers:
        target_color = marker['color'] # BGR
        dist = color_distance(avg_color, target_color)

        if dist < min_dist:
            min_dist = dist
            matched_name = marker['name']

    if min_dist < threshold:
        return True, matched_name

    return False, None

def process_video(video_path, config, markers):
    """Process a single video file"""
    video_name = Path(video_path).stem
    images_dir = Path('images') / video_name
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting to process video: {video_path}")
    logger.info(f"Corrected image output directory: {images_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

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

    logger.info(f"Output size: {output_width}x{output_height} (ratio {output_ratio[0]}:{output_ratio[1]})")

    # Get reference frame and detect screen trapezoid
    ref_frame, actual_ref_index = get_reference_frame(cap, ref_frame_index)
    if ref_frame is None:
        logger.error("Cannot read reference frame")
        cap.release()
        return

    # Rotate reference frame
    if rotation_code is not None:
        ref_frame = cv2.rotate(ref_frame, rotation_code)

    # Detect screen area from reference frame
    screen_pts = detect_screen_trapezoid(ref_frame)
    if screen_pts is None:
        logger.error(f"Cannot detect screen area from reference frame {actual_ref_index}")
        cap.release()
        return

    logger.info(f"Detected screen area from reference frame {actual_ref_index}")
    logger.info(f"Screen corners: {screen_pts.tolist()}")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate frame
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)

            # Use fixed screen corners for perspective transform correction
            corrected = perspective_transform(frame, screen_pts, output_size)

            # Save corrected image to images directory
            corrected_filename = f"frame_{frame_count:06d}.jpg"
            corrected_path = images_dir / corrected_filename
            cv2.imwrite(str(corrected_path), corrected)

            frame_count += 1
            saved_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...", end='\r')

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        cap.release()
        print()  # New line
        logger.info(f"Processing complete. Total frames: {frame_count}, Corrected frames saved: {saved_count}")

def main():
    parser = argparse.ArgumentParser(description='Process video files and extract frames')
    parser.add_argument('video_file', nargs='?', help='Video file path to process')
    args = parser.parse_args()

    # Load default config
    try:
        config_path = str(get_resource_path("default.yaml", "configs"))
    except FileNotFoundError:
        config_path = "configs/default.yaml"

    config = load_yaml(config_path)
    if not config:
        return

    # Load layout
    device_name = config.get('device')
    layout_path = None

    if device_name:
        try:
            layout_path = str(get_resource_path(f"{device_name}.yaml", "layouts"))
        except FileNotFoundError:
            pass

    if not layout_path:
        # Fallback to manual layout_file in config or default jupiter
        layout_file = config.get('layout_file', 'jupiter.yaml')
        # If just a name
        if '/' not in layout_file and '\\' not in layout_file:
             try:
                layout_path = str(get_resource_path(layout_file, "layouts"))
             except FileNotFoundError:
                layout_path = f"layouts/{layout_file}"
        else:
             layout_path = layout_file

    if os.path.exists(layout_path):
        layout_config = load_yaml(layout_path)
    else:
        logger.error(f"Layout file not found: {layout_path}")
        return

    if not layout_config:
        logger.error("Cannot load layout file, unable to get color definitions")
        return

    markers = get_markers_from_layout(layout_config)
    logger.info(f"Loaded {len(markers)} color marker definitions")

    if args.video_file:
        if os.path.exists(args.video_file):
            process_video(args.video_file, config, markers)
        else:
            logger.error(f"File does not exist: {args.video_file}")
    else:
        # Find all mp4 files in recording directory
        video_files = glob.glob('recording/*.mp4')

        if not video_files:
            logger.warning("No mp4 files found in recording directory")
            return

        for video_file in video_files:
            process_video(video_file, config, markers)

if __name__ == "__main__":
    main()
