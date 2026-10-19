# Real-Time 3D Nose Tracking with Orbbec Femto Bolt

A real-time 3D nose-tracking system using the **Orbbec Femto Bolt** depth camera, MediaPipe Pose Landmarker, OpenCV, and OSC (Open Sound Control).

The application detects a person's nose in the RGB image, retrieves the corresponding depth measurement, reconstructs its 3D position relative to the camera, applies exponential moving average (EMA) smoothing, and streams the coordinates over OSC for integration with interactive installations, audiovisual systems, and real-time applications.

## Features

- **Real-time pose detection** using MediaPipe Pose Landmarker.
- **RGB and depth acquisition** from the Orbbec Femto Bolt.
- **3D position reconstruction** using camera intrinsics and depth measurements.
- **Coordinate streaming over OSC** for integration with external software.
- **EMA smoothing** to reduce jitter in the tracked position.
- **Asynchronous frame acquisition** using a dedicated thread and a single-frame queue to prioritize the latest available frame.
- **Performance logging** to monitor processing times.
- **Timestamped OSC logging** for debugging and synchronization analysis.

## Hardware Requirements

- Orbbec Femto Bolt depth camera.
- A computer capable of running Python, OpenCV, and MediaPipe.
- A USB connection compatible with the camera's requirements.

## Software Requirements

- Python 3
- OpenCV
- NumPy
- MediaPipe Tasks
- PyOrbbecSDK
- python-osc
- Orbbec SDK dependencies and camera drivers, as required by your operating system.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install opencv-python numpy mediapipe python-osc
```

Install and configure **PyOrbbecSDK** according to the official Orbbec SDK documentation and the instructions for your operating system.

Make sure the Orbbec SDK and Python bindings are compatible with your Python version.

### 4. Download the MediaPipe model

Download the Pose Landmarker Lite model and place it in the project directory.

The application expects the following file:

```text
pose_landmarker_lite.task
```

The model path can be changed using the `MODEL_PATH` variable in the script.

### 5. Check the camera configuration

Connect the Orbbec Femto Bolt and ensure that the camera is detected by the Orbbec SDK.

The application requests an RGB stream at 1280 × 720 and 30 FPS. The depth stream uses the camera's default depth profile.

The script also imports a helper function from `utils.py`:

```python
from utils import frame_to_bgr_image
```

Ensure that this utility module is available in the project and compatible with your installed Orbbec SDK version.

## Usage

Run the tracking script:

```bash
python main.py
```

Replace `main.py` with the actual name of your Python script if necessary.

A window displays the RGB camera feed with the detected nose marked by a green circle. The estimated 3D coordinates are displayed next to the detection.

Press **Q** or **Esc** to exit.

## How It Works

The processing pipeline consists of the following steps:

1. **Frame acquisition** — RGB and depth frames are retrieved from the Orbbec Femto Bolt in a dedicated acquisition thread.
2. **Frame alignment** — The depth stream is aligned with the RGB stream so that depth measurements can be sampled at RGB pixel coordinates.
3. **Pose detection** — MediaPipe Pose Landmarker identifies body landmarks in the RGB image.
4. **Nose localization** — The nose landmark is converted from normalized image coordinates to pixel coordinates.
5. **Depth sampling** — The corresponding depth value is retrieved from the aligned depth frame.
6. **3D reconstruction** — The pixel coordinates and depth measurement are converted into 3D camera-space coordinates using the RGB camera intrinsics.
7. **Smoothing** — An exponential moving average reduces fluctuations in the estimated position.
8. **OSC transmission** — The smoothed coordinates are transmitted to an OSC receiver over UDP.

## 3D Coordinates and Units

The reconstructed position is calculated using a pinhole-camera model:

```python
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z
Z = depth
```

Where:

- `u`, `v` are the nose pixel coordinates.
- `fx`, `fy` are the camera focal lengths in pixel units.
- `cx`, `cy` are the principal-point coordinates in pixels.
- `Z` is the measured depth.

The script converts the resulting coordinates from millimeters to centimeters by dividing each coordinate by 10.

**The coordinates sent over OSC are therefore expressed in centimeters**, provided that the depth scale returned by the SDK converts the depth values to millimeters.

The coordinates are relative to the camera's optical coordinate system, not absolute world coordinates.

- `X`: horizontal position.
- `Y`: vertical position.
- `Z`: distance from the camera.

The coordinate signs depend on the camera coordinate convention and the position of the nose relative to the optical axis.

### Camera Intrinsics

The script initializes the RGB camera intrinsics using predefined values for a native 3840 × 2160 resolution and scales them to the actual RGB image resolution.

These values must correspond to the camera's calibration and image mode. For accurate measurements, use the correct intrinsic parameters for the active camera configuration.

## OSC Output

The application sends the nose position to an OSC receiver using UDP.

Default configuration:

```python
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
OSC_ADDRESS = "/nose"
```

The OSC message contains three floating-point values:

```text
/nose X Y Z
```

Example:

```text
/nose 2.4 -8.1 65.7
```

In this example, the nose is located 2.4 cm along X, -8.1 cm along Y, and 65.7 cm from the camera along Z.

The receiver can be configured in any OSC-compatible environment, such as TouchDesigner, Max/MSP, Pure Data, or a custom Python application.

To send data to another computer, change `OSC_IP` to the receiver's IP address and ensure that UDP traffic is permitted by the network and firewall configuration.

## Configuration

The following parameters can be modified in the script:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `pose_landmarker_lite.task` | MediaPipe model file |
| `TARGET_WIDTH` | `1280` | RGB image width |
| `TARGET_HEIGHT` | `720` | RGB image height |
| `TARGET_FPS` | `30` | Requested RGB frame rate |
| `OSC_IP` | `127.0.0.1` | OSC receiver IP address |
| `OSC_PORT` | `9000` | OSC receiver UDP port |
| `OSC_ADDRESS` | `/nose` | OSC address pattern |
| `EMA_ALPHA` | `0.4` | Smoothing coefficient |
| `MIN_DEPTH` | `20` | Minimum accepted depth value |
| `MAX_DEPTH` | `10000` | Maximum accepted depth value |

The smoothing coefficient controls the balance between responsiveness and stability. Lower values provide stronger smoothing but introduce more lag, while higher values respond faster to movement.

The depth thresholds and their units must be consistent with the depth scale returned by the SDK.

## Logging and Performance Monitoring

The application generates two log files:

- `performance.log` — records processing durations and total frame processing times.
- `timestamp.txt` — records timestamps and the coordinates transmitted over OSC.

These files can be used to evaluate latency, inspect tracking behavior, and analyze the timing of outgoing OSC messages.

## Limitations and Considerations

- The system tracks the nose landmark provided by MediaPipe; it is not a dedicated facial landmark tracker.
- Depth measurements can become unreliable when the nose is occluded, near the edge of the image, or affected by depth-sensor noise.
- Sampling depth at a single pixel can produce unstable measurements. Neighborhood-based sampling or median filtering may improve robustness.
- The depth frame must be correctly aligned with the RGB frame before sampling.
- The predefined camera intrinsics should be verified for the selected RGB resolution and camera configuration.
- The current implementation uses camera-relative coordinates. A world-space coordinate system would require an additional transformation and a defined world origin.
- OSC transmission uses UDP, which does not guarantee delivery of every message.

## Possible Applications

- Interactive installations and real-time audiovisual systems.
- Motion-reactive visuals.
- Spatial interaction and proximity-based control.
- Gesture and body-tracking experiments.
- Real-time control of external software through OSC.
- Research and prototyping involving RGB-D cameras.

## License

Specify the license for this repository in the `LICENSE` file. If the project is distributed publicly, ensure that the licenses of all third-party libraries and model assets are respected.

## Acknowledgments

This project uses:

- [Orbbec SDK](https://github.com/orbbec/OrbbecSDK)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [OpenCV](https://opencv.org/)
- [python-osc](https://github.com/attwad/python-osc)

