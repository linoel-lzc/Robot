import time
import serial
import serial.tools.list_ports
import logging
import yaml
import math
import platform
import numpy as np

logger = logging.getLogger(__name__)

class RobotArmController:
    def __init__(self, port=None, baudrate=93450, timeout=60, layout_file=None, config_file=None):
        """
        Initialize the RobotArmController.

        :param port: The serial port to connect to (e.g., '/dev/ttyUSB0' on Linux or '/dev/cu.usbserial-*' on macOS). If None, attempts to auto-detect.
        :param baudrate: The baud rate for the serial connection. Default is 93450.
        :param timeout: Read timeout in seconds.
        :param layout_file: Path to layout.yaml file.
        :param config_file: Path to config file (e.g. configs/default.yaml) for calibration.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

        self.layout = None
        self.buttons = {}
        self.colors = {}  # Color definitions from layout
        self.transform_matrix = None  # 2x3 affine matrix [[a, b, tx], [c, d, ty]]
        self.click_params = {'press_depth': 10, 'press_duration': 0.2, 'lift_height': 0}
        self.speed = {'xy': 8000, 'z': 1000}  # Default speed config (mm/min)
        self.capture = None  # Optional Capture instance for video marking
        self.config = {} # Store full config

        # Load config first to get potential layout_file path
        if config_file:
            self.load_config(config_file, apply_calibration=False)

        # Determine layout file path: prefer parameter, then config file path
        target_layout_file = layout_file
        if not target_layout_file and self.config.get('layout_file'):
            target_layout_file = self.config.get('layout_file')

        # Only load layout if not already loaded (avoid duplicate loading after load_config_from_dict)
        if target_layout_file and not self.layout:
            self.load_layout(target_layout_file)

        # Layout loaded, now apply calibration if we have config
        if self.layout and self.config:
            self.apply_calibration()
        elif not layout_file and not config_file:
            logger.warning("No layout file or config file provided. Controller initialized without layout.")

        if self.port is None:
            self.port = self._detect_port()

    def _detect_port(self):
        """Detect the first available serial port based on the operating system."""
        ports = list(serial.tools.list_ports.comports())
        logger.info(f"Found {len(ports)} ports.")
        if not ports:
            raise Exception("No serial ports found!")

        # Log all available ports for debugging
        for idx, port in enumerate(ports):
            logger.debug(f"  [{idx}] {port.device} - {port.description}")

        # Get the operating system
        os_type = platform.system()

        # Filter ports based on OS
        if os_type == "Darwin":  # macOS
            # On macOS, prefer USB serial devices:
            # 1. /dev/cu.usbserial-* (FTDI and other USB-to-serial adapters)
            # 2. /dev/cu.usbmodem-* (USB CDC/ACM devices)
            # 3. Other /dev/cu.* ports (but exclude Bluetooth and debug)
            usb_serial_ports = [p for p in ports if 'usbserial' in p.device.lower()]
            if usb_serial_ports:
                selected_port = usb_serial_ports[0].device
                logger.info(f"[macOS] Auto-selected USB serial port: {selected_port}")
                return selected_port

            usb_modem_ports = [p for p in ports if 'usbmodem' in p.device.lower()]
            if usb_modem_ports:
                selected_port = usb_modem_ports[0].device
                logger.info(f"[macOS] Auto-selected USB modem port: {selected_port}")
                return selected_port

            # Filter out known non-device ports
            filtered_ports = [p for p in ports
                            if p.device.startswith('/dev/cu.')
                            and 'Bluetooth' not in p.device
                            and 'debug' not in p.device.lower()]
            if filtered_ports:
                selected_port = filtered_ports[0].device
                logger.info(f"[macOS] Auto-selected port: {selected_port}")
                return selected_port

        elif os_type == "Linux":
            # On Linux, prefer /dev/ttyUSB* or /dev/ttyACM* ports
            usb_ports = [p for p in ports if '/dev/ttyUSB' in p.device or '/dev/ttyACM' in p.device]
            if usb_ports:
                selected_port = usb_ports[0].device
                logger.info(f"[Linux] Auto-selected port: {selected_port}")
                return selected_port

        elif os_type == "Windows":
            # On Windows, select first USB serial port (COMx)
            usb_ports = [p for p in ports if 'USB' in p.description.upper()]
            if usb_ports:
                selected_port = usb_ports[0].device
                logger.info(f"[Windows] Auto-selected USB serial port: {selected_port}")
                return selected_port

        # Fallback: select the first available port
        first_port = ports[0].device
        logger.warning(f"[{os_type}] No USB serial port found, using fallback: {first_port}")
        return first_port

    def connect(self):
        """Establish the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            return

        logger.info(f"Connecting to {self.port} at {self.baudrate} baud...")
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logger.info(f"Connected to {self.serial_conn.name}")
            # Important: Wait for robot to be ready
            logger.info("Waiting for robot arm to initialize...")
            time.sleep(2)
        except serial.SerialException as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Connection closed.")

    def _send_command(self, command, wait_after=0.01):
        """Send a command string to the robot arm."""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise Exception("Not connected to robot arm.")

        full_command = f"{command}\r"
        logger.debug(f"Sending command: {full_command.strip()}")
        self.serial_conn.write(full_command.encode())

        if wait_after > 0:
            time.sleep(wait_after)

    def set_feed_rate(self, speed, wait=0.01):
        """Set the feed rate (speed) for subsequent moves.

        :param speed: Feed rate in mm/min (100-10000).
        :param wait: Wait time after sending command.
        """
        cmd = f"G1F{speed}"
        self._send_command(cmd, wait_after=wait)

    def move_to(self, x, y, speed=None, wait=0.01):
        """Move the arm to specified X, Y coordinates.

        :param x: Target X position.
        :param y: Target Y position.
        :param speed: Feed rate in mm/min (100-10000). If None, uses xy speed from config.
        :param wait: Wait time after sending command.
        """
        if speed is None:
            speed = self.speed.get('xy', 8000)
        self.set_feed_rate(speed, wait=wait)
        cmd = f"G1X{x}Y{y}"
        self._send_command(cmd, wait_after=wait)

    def move_z(self, z, speed=None, wait=0.01):
        """Move the Z axis.

        :param z: Target Z position.
        :param speed: Feed rate in mm/min (100-10000). If None, uses z speed from config.
        :param wait: Wait time after sending command.
        """
        if speed is None:
            speed = self.speed.get('z', 1000)
        self.set_feed_rate(speed, wait=wait)
        cmd = f"G1Z{z}"
        self._send_command(cmd, wait_after=wait)

    def press(self, depth=10, duration=None, speed=None):
        """
        Convenience method to press down (Z axis).

        :param depth: Z depth to press (default 10).
        :param duration: If provided, waits this many seconds then lifts (Z=0).
        :param speed: Feed rate in mm/min (100-10000). If None, uses z speed from config.
        """
        self.move_z(depth, speed=speed)

        if duration is not None:
            logger.info(f"Pressed down for {duration} seconds")
            time.sleep(duration)
            self.lift(speed=speed)


    def lift(self, speed=None, wait=0.01):
        """Convenience method to lift (Z=0).

        :param speed: Feed rate in mm/min (100-10000). If None, uses z speed from config.
        :param wait: Wait time after sending command.
        """
        self.move_z(0, speed=speed, wait=wait)

    def home(self, speed=None, wait=0.01):
        """Return to home position (X0 Y0).

        :param speed: Feed rate in mm/min. If None, uses xy speed from config.
        :param wait: Wait time after sending command.
        """
        if speed is None:
            speed = self.speed.get('xy', 8000)
        self.set_feed_rate(speed, wait=wait)
        self._send_command("G1X0Y0", wait_after=wait)

    # --- Layout and Calibration Methods ---

    def load_layout(self, filepath):
        """Load button layout from yaml file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        self.layout = data.get('device', {})
        self.buttons = {btn['name']: btn for btn in self.layout.get('buttons', [])}
        self.colors = self.layout.get('colors', {})
        logger.info(f"Loaded layout with {len(self.buttons)} buttons and {len(self.colors)} colors.")

    def set_capture(self, capture):
        """Set the Capture instance for video marking.

        :param capture: A Capture instance for video recording and marking.
        """
        self.capture = capture

        # Try to apply marker_region from config (prefer capture, fallback to video_processing for backward compatibility)
        if self.config:
            capture_config = self.config.get('capture', {})
            vp_config = self.config.get('video_processing', {})
            marker_region = capture_config.get('marker_region') or vp_config.get('marker_region')
            if marker_region and hasattr(self.capture, 'set_marker_region'):
                self.capture.set_marker_region(marker_region)

        logger.info("Capture instance attached to controller.")

    def load_config(self, filepath, apply_calibration=True):
        """Load calibration config from yaml file."""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        self.load_config_from_dict(config, apply_calibration=apply_calibration)

    def load_config_from_dict(self, config, apply_calibration=True):
        """Load calibration config from dictionary.

        :param config: Configuration dictionary.
        :param apply_calibration: Whether to apply calibration after loading.
        """
        self.config = config  # Save full config

        # Load layout
        # 1. First check 'device' field, auto-load corresponding built-in layout
        device_name = config.get('device')
        layout_file = config.get('layout_file')

        target_layout_file = None

        if device_name:
            # Try to load built-in layout
            from .utils import get_resource_path
            try:
                target_layout_file = get_resource_path(f"{device_name}.yaml", "layouts")
                logger.info(f"Using device layout for '{device_name}': {target_layout_file}")
            except FileNotFoundError:
                logger.warning(f"Layout for device '{device_name}' not found.")

        # 2. If no device or not found, fallback to layout_file (backward compatibility/manual specification)
        if not target_layout_file and layout_file:
            target_layout_file = layout_file

        if target_layout_file and not self.layout:
            self.load_layout(target_layout_file)

        # Load click parameters
        if 'click_params' in config:
            self.click_params.update(config['click_params'])

        # Load speed configuration
        if 'speed' in config:
            self.speed.update(config['speed'])
            logger.info(f"Speed config: XY={self.speed['xy']} mm/min, Z={self.speed['z']} mm/min")

        # If capture already exists, update its config (prefer capture, fallback to video_processing for backward compatibility)
        if self.capture and hasattr(self.capture, 'set_marker_region'):
            capture_config = self.config.get('capture', {})
            vp_config = self.config.get('video_processing', {})
            marker_region = capture_config.get('marker_region') or vp_config.get('marker_region')
            if marker_region:
                self.capture.set_marker_region(marker_region)

        # Perform calibration if points are provided
        if apply_calibration:
            self.apply_calibration()

    def apply_calibration(self):
        """Apply calibration from current config. Requires exactly 3 calibration points."""
        calib = self.config.get('calibration', {})

        # Validate: must have exactly 3 calibration points
        required_points = ['point1', 'point2', 'point3']
        missing_points = [p for p in required_points if p not in calib]
        if missing_points:
            raise ValueError(
                f"Calibration requires 3 points. Missing: {missing_points}. "
                f"Please configure point1, point2, and point3 in your config file."
            )

        # Check if layout is loaded
        if not self.buttons:
            logger.warning("Cannot apply calibration: Layout not loaded yet.")
            return

        self._calibrate(calib['point1'], calib['point2'], calib['point3'])

    def _calibrate(self, p1_conf, p2_conf, p3_conf):
        """
        Calculate affine transformation using 3 points.
        Supports rotation, scaling, and translation.

        [rx]   [a b] [lx]   [tx]
        [ry] = [c d] [ly] + [ty]

        Stored as transform_matrix = [[a, b, tx], [c, d, ty]]
        """
        btn_names = [p1_conf['name'], p2_conf['name'], p3_conf['name']]
        for name in btn_names:
            if name not in self.buttons:
                raise ValueError(f"Calibration button '{name}' not found in layout.")

        # Layout coordinates (source)
        l1 = self.buttons[btn_names[0]]['center']
        l2 = self.buttons[btn_names[1]]['center']
        l3 = self.buttons[btn_names[2]]['center']

        # Robot coordinates (target)
        r1 = p1_conf['robot_coords']
        r2 = p2_conf['robot_coords']
        r3 = p3_conf['robot_coords']

        # Build linear system to solve for affine parameters
        # For X: r_x = a*l_x + b*l_y + tx
        # For Y: r_y = c*l_x + d*l_y + ty
        A = np.array([
            [l1[0], l1[1], 1],
            [l2[0], l2[1], 1],
            [l3[0], l3[1], 1]
        ], dtype=np.float64)

        bx = np.array([r1[0], r2[0], r3[0]], dtype=np.float64)
        by = np.array([r1[1], r2[1], r3[1]], dtype=np.float64)

        # Check if points are collinear (matrix would be singular)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            raise ValueError(
                "Calibration points are collinear (on the same line). "
                "Please choose 3 points that form a triangle."
            )

        # Solve for [a, b, tx] and [c, d, ty]
        x_params = np.linalg.solve(A, bx)  # [a, b, tx]
        y_params = np.linalg.solve(A, by)  # [c, d, ty]

        self.transform_matrix = np.array([x_params, y_params])

        # Log transformation details
        a, b, tx = x_params
        c, d, ty = y_params
        # Calculate rotation angle from matrix (approximate, assumes uniform scaling)
        angle_rad = math.atan2(c, a)
        angle_deg = math.degrees(angle_rad)
        scale = math.sqrt(a*a + c*c)

        logger.info(
            f"Calibration complete (3-point affine): "
            f"Rotation≈{angle_deg:.1f}°, Scale≈{scale:.3f}, "
            f"Translation=({tx:.2f}, {ty:.2f})"
        )

    def _layout_to_robot(self, layout_x, layout_y):
        """Convert layout coordinates to robot coordinates using affine transformation."""
        if self.transform_matrix is None:
            logger.warning("No calibration loaded. Using 1:1 mapping.")
            return layout_x, layout_y

        # transform_matrix = [[a, b, tx], [c, d, ty]]
        a, b, tx = self.transform_matrix[0]
        c, d, ty = self.transform_matrix[1]
        rx = a * layout_x + b * layout_y + tx
        ry = c * layout_x + d * layout_y + ty
        return rx, ry

    def click_button(self, button_name, on_press_callback=None, on_lift_callback=None, enable_capture=True):
        """
        Click a button by its name defined in the layout.

        :param button_name: Name of the button (e.g. "Key_1").
        :param on_press_callback: Optional callback function called immediately after pressing down (for numeric key marking).
        :param on_lift_callback: Optional callback function called immediately after lifting up (for green key marking).
        :param enable_capture: Whether to use automatic capture marking from layout config (default: True).
        """
        if button_name not in self.buttons:
            raise ValueError(f"Button '{button_name}' not found in layout.")

        btn = self.buttons[button_name]
        lx, ly = btn['center']

        rx, ry = self._layout_to_robot(lx, ly)
        logger.info(f"Clicking '{button_name}'")
        logger.debug(f"Clicking '{button_name}' at Robot Coords: ({rx:.2f}, {ry:.2f})")

        # Move to target position
        self.move_to(rx, ry)

        # Use parameters from config or defaults
        depth = self.click_params.get('press_depth', 10)
        duration = self.click_params.get('press_duration', 0.2)
        lift_h = self.click_params.get('lift_height', 0)

        # Get button's capture config
        capture_config = btn.get('capture', {})
        mark_on = capture_config.get('mark_on', None)

        # Press down
        self.move_z(depth)

        # Post-press handling
        # Prefer external callback, otherwise use auto-marking
        if on_press_callback:
            on_press_callback()
        elif enable_capture and self.capture and mark_on == 'press':
            self._mark_capture(button_name, capture_config)

        # Hold
        if duration is not None:
            logger.debug(f"Holding for {duration} seconds")
            time.sleep(duration)

        # Lift up
        self.lift()

        # Post-lift handling
        # Prefer external callback, otherwise use auto-marking
        if on_lift_callback:
            on_lift_callback()
        elif enable_capture and self.capture and mark_on == 'lift':
            self._mark_capture(button_name, capture_config)

        if lift_h != 0:
            self.move_z(lift_h)

    def _mark_capture(self, button_name, capture_config):
        """Mark video frame based on config

        :param button_name: Button name
        :param capture_config: Config dict containing color and position
        """
        if not self.capture:
            return

        # Get color config
        color_ref = capture_config.get('color')
        if not color_ref:
            return

        # If string, look up in colors dict; otherwise use BGR array directly
        if isinstance(color_ref, str):
            color = self.colors.get(color_ref)
            if not color:
                logger.warning(f"Color '{color_ref}' not found in color definitions.")
                return
        else:
            color = color_ref

        # Ensure color is tuple format (OpenCV compatibility)
        if isinstance(color, list):
            color = tuple(color)

        # Mark frame
        self.capture.mark_frame(color, button_name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
