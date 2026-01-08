# RobotArm v1

RobotArm is a Python package for controlling a robotic arm to execute key sequences and performing visual verification.

Repository: `git@sqbu-github.cisco.com:Walle/RobotArm.git`

## Features

- **Performer**: Automatically controls the robotic arm to execute predefined key sequences (Melody/Score).
- **Observer**: Analyzes video recordings, visually identifies key events, and verifies execution results.
- **Configuration Management**:
    - **Configs**: System-level configurations (serial port, camera, etc.), supporting built-in defaults or user customization.
    - **Layouts**: Device key layouts, automatically loaded by the system based on `device`, no user configuration required.
    - **Scores**: User-defined execution scripts (Melody).
- **CLI Tool**: Provides a unified command-line tool `robotarm`.

## Installation and Environment Setup

**Prerequisites**:
1. Python >= 3.10
2. SSH Key configured (ensure access to repository `git@sqbu-github.cisco.com:Walle/RobotArm.git`)
3. `uv` (recommended) or `pip` installed

### Step 1: Create an Independent Environment (Recommended)

Using `uv`:
```bash
# 1. Create virtual environment
uv venv

# 2. Activate environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### Step 2: Install RobotArm Package

You can install directly from the Git repository. The URL must use the `git+ssh://` protocol.

**Using uv (Recommended):**
```bash
uv pip install "git+ssh://git@sqbu-github.cisco.com/Walle/RobotArm.git"
```

**Using pip:**
```bash
pip install "git+ssh://git@sqbu-github.cisco.com/Walle/RobotArm.git"
```

*(Alternative) Install from Source:*
```bash
git clone git@sqbu-github.cisco.com:Walle/RobotArm.git
cd RobotArm
uv sync  # or pip install -e .
```

## Configuration Guide (Configs & Scores)

RobotArm uses YAML files for configuration.

### 1. System Configuration (Configs)

System configuration includes robotic arm connection parameters, camera settings, VLM model configuration, etc.

**Default vs Custom Configuration:**

*   **Default Configuration**: If you do not provide any configuration, or if you reference `"default.yaml"` in your Score file, the system will automatically load the built-in `robotarm/resources/configs/default.yaml`. This is usually suitable for standard development environments.
*   **Custom Configuration**: If you need to modify the serial port, adjust calibration points, or change camera settings, please create a new YAML file in the current directory (e.g., `my_config.yaml`) and reference it in the Score file.

**Config YAML Structure Reference:**

```yaml
# 1. Layout Configuration
device: "jupiter"  # Specify device name, system automatically loads corresponding layout file (robotarm/resources/layouts/<device>.yaml)

# 2. Calibration Point Configuration (Corresponds to keys in Layout)
calibration:
  point1:
    name: "Key_Green"       # Key name in Layout
    robot_coords: [21.25, 96] # Actual robot coordinates (X, Y)
  point2:
    name: "Key_Red"
    robot_coords: [54, 96]

# 3. Click Parameters (Control Z-axis)
click_params:
  press_depth: 2      # Press depth (unit: mm, relative to Z=0)
  press_duration: 0.4 # Press hold time (seconds)
  lift_height: 0      # Lift height (unit: mm)

# 4. Movement Speed (mm/min)
speed:
  xy: 10000 # X/Y plane movement speed
  z: 1000   # Z-axis vertical movement speed

# 5. Video Capture Configuration
capture:
  backend: "opencv"   # Backend: opencv, ffmpeg, pipe
  camera_name: "Cisco Desk Camera 4K"
  width: 1920
  height: 1080
  fps: 30
  output_dir: "recording" # Video recording save directory
  # Marker Region (Draw/Detect color square in top-left corner of frame)
  marker_region:
    x: 0
    y: 0
    width: 100
    height: 100

# 6. Video Processing Configuration (Used by Observer)
video_processing:
  rotate: 270 # Video rotation angle: 0, 90, 180, 270
  color_threshold: 50 # Color match threshold
  # Screen Correction (For correcting trapezoidal screen to rectangle before VLM recognition)
  screen_correction:
    reference_frame: -1 # Reference frame index (-1 is the last frame)
    output_ratio: [3, 4] # Output aspect ratio
    output_height: 640

# 7. VLM Configuration (Visual Language Model)
vlm:
  base_url: "http://your-vlm-server/v1/" # VLM API Address
  api_key: "your-api-key"
  model: "ministral-3-8b-instruct"
  temperature: 0.0
```

### 2. Execution Scripts (Scores / Melody)

The Score file defines the sequence of actions the robotic arm needs to execute. This is a file the user must provide.

**Score YAML Structure Reference:**

```yaml
name: "Dialing Test"
description: "Test dialing flow"

# Reference system configuration file (can be built-in default.yaml or custom path)
config_file: "configs/default.yaml"

# (Optional) Override partial settings in Config
config:
  device: "jupiter"

# (Optional) Recording Configuration
recording:
  prefix: "my_test" # Video filename prefix

# (Optional) Observer Configuration
observer:
  save_frames: true      # Whether to save original frames
  save_corrected: true   # Whether to save corrected images
  compare: true          # Whether to compare detection results with action sequence
  debug: false

# Action Sequence
actions:
  # Action Type: home
  - type: home
    comment: "Return to home position"

  # Action Type: button
  - type: button
    name: "Key_1"       # Must be a key name existing in Layout
    wait_before: 0.5    # Wait before execution (seconds)
    wait_after: 1       # Wait after execution (seconds)
    comment: "Press Key 1"
    # (Optional) VLM Verification Question: Verify screen state after operation
    ask: "Is number 1 displayed on the screen?"

  # Action Type: wait
  - type: wait
    duration: 2         # Wait time (seconds)
  
  # Action Type: lift (Lift only)
  - type: lift

  # Action Type: move (Move only)
  - type: move
    x: 100
    y: 100
```

## CLI Usage

### 1. Execute Sequence (Performer)

```bash
# Execute and record video (Default)
robotarm perform -m my_score.yaml

# Execute only, no recording
robotarm perform -m my_score.yaml --no-record
```

### 2. Verify Results (Observer)

```bash
# Analyze video and verify results based on my_score.yaml
robotarm observe recording/video.mp4 -m my_score.yaml
```

## Python API Usage Example

You can also integrate directly in Python code:

### 1. Performer Call

```python
from robotarm import RobotArmController, run_melody
from robotarm.utils import get_resource_path
import yaml

# 1. Initialize controller
controller = RobotArmController()

# 2. Load Score configuration (Load built-in example here)
score_path = get_resource_path("dial_8890.yaml", "scores")
with open(score_path, "r") as f:
    melody = yaml.safe_load(f)

# 3. Execute
try:
    with controller:
        run_melody(controller, melody)
except KeyboardInterrupt:
    print("Stopped by user")
```

### 2. Observer Call

```python
from robotarm import observe_video, get_resource_path
import yaml

# Video path
video_path = "recording/my_video.mp4"

# 1. Load Melody configuration (Contains expected actions)
score_path = get_resource_path("dial_8890.yaml", "scores")
with open(score_path, "r") as f:
    melody = yaml.safe_load(f)
    
# 2. Load System Configuration (Config)
# Note: If melody already references config_file, Observer loads it automatically.
# But if you need to manually pass the config object, you can do this:
config_path = get_resource_path("default.yaml", "configs")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 3. Load Layout Configuration (Layout)
# Usually 'device' field in Config handles Layout loading automatically, but can also be manually specified
layout_path = get_resource_path("jupiter.yaml", "layouts")
with open(layout_path, "r") as f:
    layout = yaml.safe_load(f)

# 4. Call Observer for Analysis
# events will return list of detected key events
events = observe_video(
    video_path=video_path,
    config=config,
    layout_config=layout,
    melody=melody,
    save_frames=True,   # Whether to save frames
    debug=False
)

# Print results
for event in events:
    print(f"Detected {event.key_name} at {event.timestamp_ms}ms")
```

## Directory Structure

- `robotarm/`: Core code library
    - `resources/`: Built-in resources (Default Configs, Layouts)
- `tests/`: Test code
- `configs/`: (Optional) User directory for custom configurations
- `scores/`: (Optional) User directory for Score files
- `logs/`: Runtime log output
- `recording/`: Video recording output
