import sys
print(sys.executable)

import cv2
import numpy as np
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pyorbbecsdk import *
from utils import frame_to_bgr_image

from pythonosc import udp_client


# =================================
# CONFIG
# =================================

MODEL_PATH = "pose_landmarker_heavy.task"

ESC_KEY = 27
PRINT_INTERVAL = 0.5

MIN_DEPTH = 20
MAX_DEPTH = 10000

COLOR_FX_NATIVE = 2242.987061
COLOR_FY_NATIVE = 2241.741699
COLOR_CX_NATIVE = 1893.117432
COLOR_CY_NATIVE = 1091.378052
COLOR_WIDTH_NATIVE  = 3840
COLOR_HEIGHT_NATIVE = 2160

# --- OSC ---
OSC_IP   = "127.0.0.1"   # adresse destination
OSC_PORT = 9000           # port destination
OSC_ADDRESS = "/nose"     # adresse OSC du message

last_print_time = time.time()


# =================================
# OSC CLIENT
# =================================

osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
print(f"OSC client ready -> {OSC_IP}:{OSC_PORT}  address: {OSC_ADDRESS}")


# =================================
# MEDIAPIPE SETUP
# =================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(options)
print("MediaPipe Pose ready")


# =================================
# UTILS : rétroprojection 2D -> 3D
# =================================

def deproject_pixel_to_3d(u, v, z_mm, fx, fy, cx, cy):
    X = (u - cx) / fx * z_mm
    Y = (v - cy) / fy * z_mm
    Z = z_mm
    return X, Y, Z


# =================================
# MAIN
# =================================

def main():
    global last_print_time

    pipeline = Pipeline()
    config = Config()

    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        config.enable_stream(color_profile)
        print("Color stream enabled")

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        print("Depth stream enabled")

        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    except Exception as e:
        print("Config error:", e)
        return

    try:
        pipeline.start(config)
        print("Pipeline started")
    except Exception as e:
        print("Start error:", e)
        return

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    cv2.namedWindow("RGB Viewer",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Viewer",   1920, 1080)
    cv2.resizeWindow("Depth Viewer", 1920, 1080)
    cv2.moveWindow("RGB Viewer",   50,   50)
    cv2.moveWindow("Depth Viewer", 2000, 50)

    fx = fy = cx = cy = None

    while True:
        try:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            frames = align_filter.process(frames)
            if not frames:
                continue

            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert color frame")
                continue

            height, width, _ = color_image.shape

            if fx is None:
                scale_x = width  / COLOR_WIDTH_NATIVE
                scale_y = height / COLOR_HEIGHT_NATIVE
                fx = COLOR_FX_NATIVE * scale_x
                fy = COLOR_FY_NATIVE * scale_y
                cx = COLOR_CX_NATIVE * scale_x
                cy = COLOR_CY_NATIVE * scale_y
                print(f"Effective color resolution: {width}x{height}")
                print(f"Scaled intrinsics -> fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)

            if depth_frame.get_format() != OBFormat.Y16:
                print("Depth format is not Y16")
                continue

            depth_data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape((depth_frame.get_height(), depth_frame.get_width()))

            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

            depth_data = np.where(
                (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH),
                depth_data, 0
            )

            current_time = time.time()

            if detection_result.pose_landmarks:
                nose = detection_result.pose_landmarks[0][0]

                nose_u = int(nose.x * width)
                nose_v = int(nose.y * height)
                nose_u = max(0, min(nose_u, width  - 1))
                nose_v = max(0, min(nose_v, height - 1))

                nose_z_mm = depth_data[nose_v, nose_u]

                if nose_z_mm > 0:
                    X, Y, Z = deproject_pixel_to_3d(
                        nose_u, nose_v, nose_z_mm, fx, fy, cx, cy
                    )

                    # --- Envoi OSC ---
                    osc_client.send_message(OSC_ADDRESS, [float(X), float(Y), float(Z)])

                    if current_time - last_print_time >= PRINT_INTERVAL:
                        print(
                            f"Nose -> pixel=({nose_u}, {nose_v}) | "
                            f"3D (mm) X={X:+.1f}  Y={Y:+.1f}  Z={Z:.1f}  "
                            f"-> OSC sent"
                        )
                        last_print_time = current_time

                    label = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"
                    cv2.circle(color_image, (nose_u, nose_v), 8, (0, 255, 0), -1)
                    cv2.putText(
                        color_image, label,
                        (nose_u + 12, nose_v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                else:
                    cv2.circle(color_image, (nose_u, nose_v), 8, (0, 0, 255), -1)

            depth_visual = cv2.normalize(
                depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            cv2.imshow("RGB Viewer",   color_image)
            # cv2.imshow("Depth Viewer", depth_visual)

            key = cv2.waitKey(1)
            if key == ord("q") or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Runtime error:", e)

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()