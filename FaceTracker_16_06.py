import sys
print(sys.executable)

import cv2
import numpy as np
import time
import threading
import queue
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pyorbbecsdk import *
from utils import frame_to_bgr_image

from pythonosc import udp_client


import logging
from datetime import datetime

# =================================
# PERFORMANCE LOGGER
# =================================

logging.basicConfig(
    filename="performance.log",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",
    datefmt="%H:%M:%S"
)

perf_log = logging.getLogger("performance")
# =================================
# OSC TIMESTAMP LOGGER
# =================================

osc_log = open("timestamp.txt", "a", buffering=1)

def get_timestamp_ms():
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}.{now.microsecond // 1000:03d}"

def log_step(name, start_time):
    """
    Log le temps écoulé depuis start_time
    """
    elapsed = (time.perf_counter() - start_time) * 1000
    perf_log.info(f"{name}: {elapsed:.3f} ms")
    return time.perf_counter()

# =================================
# CONFIG
# =================================

MODEL_PATH = "pose_landmarker_lite.task"

ESC_KEY = 27
PRINT_INTERVAL = 0.5

MIN_DEPTH = 20
MAX_DEPTH = 10000

# --- Résolution cible (bien plus légère que 4K) ---
# Choisir parmi ce que la caméra supporte : 1280x720, 848x480, 640x480...
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 720
TARGET_FPS    = 30

# --- OSC ---
OSC_IP      = "127.0.0.1"
OSC_PORT    = 9000
OSC_ADDRESS = "/nose"

# --- Smoothing (EMA) ---
# 0.0 = position figée, 1.0 = pas de lissage (données brutes)
# 0.3-0.5 = bon compromis fluidité / latence
EMA_ALPHA = 0.4

# Seuil de jump : si la position saute de plus de X cm en un frame, on rejette la mesure
# Protège contre les artefacts depth et les faux positifs MediaPipe en sortie de champ
MAX_JUMP_CM = 30.0

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
# UTILS
# =================================

def deproject_pixel_to_3d(u, v, z_mm, fx, fy, cx, cy):
    X = (u - cx) / fx * z_mm
    Y = (v - cy) / fy * z_mm
    Z = z_mm
    return X, Y, Z

def convert_to_m(X, Y, Z):
    return X / 10.0, Y / 10.0, Z / 10.0


# =================================
# THREAD D'ACQUISITION
# =================================

def acquisition_thread(pipeline, align_filter, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Thread dédié à l'acquisition caméra.
    Pousse uniquement la frame la plus récente dans la queue (maxsize=1 → drop automatique).
    """
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(100)  # timeout court pour rester réactif
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

            # On vide la queue avant de pousser pour toujours avoir la frame la plus fraîche
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

            frame_queue.put((color_frame, depth_frame))

        except Exception as e:
            if not stop_event.is_set():
                print(f"Acquisition error: {e}")


# =================================
# MAIN
# =================================

def main():
    global last_print_time

    pipeline = Pipeline()
    config = Config()

    color_profile = None

    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # Demander explicitement une résolution basse et un FPS élevé
        color_profile = color_profile_list.get_video_stream_profile(
            TARGET_WIDTH, TARGET_HEIGHT, OBFormat.RGB, TARGET_FPS
        )
        config.enable_stream(color_profile)
        print(f"Color stream: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()} fps")

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        print(f"Depth stream: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()} fps")

        # ANY_SITUATION : ne pas attendre la synchro parfaite des deux streams
        # Utiliser FULL_FRAME_REQUIRE seulement si la synchro temporelle est critique
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)

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
    # cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Viewer",   1920, 1080)
    # cv2.resizeWindow("Depth Viewer", 1920, 1080)
    cv2.moveWindow("RGB Viewer",   50,   50)
    # cv2.moveWindow("Depth Viewer", 2000, 50)


    # Les intrinsèques seront calculées depuis la résolution réelle de la première frame
    fx = fy = cx = cy = None

    # --- Lancer le thread d'acquisition ---
    frame_queue = queue.Queue(maxsize=1)
    stop_event  = threading.Event()

    acq_thread = threading.Thread(
        target=acquisition_thread,
        args=(pipeline, align_filter, frame_queue, stop_event),
        daemon=True
    )
    acq_thread.start()
    print("Acquisition thread started")

    frame_count = 0
    fps_timer   = time.time()

    # --- Etat du lissage EMA + derniere position valide ---
    smooth_X = smooth_Y = smooth_Z = None
    last_valid_X = last_valid_Y = last_valid_Z = None

    while True:
        try:
            frame_start = time.perf_counter()
            last_step = frame_start
            # Attendre la prochaine frame (timeout 1 s pour pouvoir checker le clavier)
            try:
                color_frame, depth_frame = frame_queue.get(timeout=1.0)
                #last_step = log_step("Frame received", last_step)
            except queue.Empty:
                key = cv2.waitKey(1)
                if key == ord("q") or key == ESC_KEY:
                    break
                continue

            color_image = frame_to_bgr_image(color_frame)
            #last_step = log_step("Frame conversion BGR", last_step)
            if color_image is None:
                continue

            height, width, _ = color_image.shape

            # Calcul des intrinsèques depuis la résolution réelle (scaling depuis native 4K)
            if fx is None:
                COLOR_FX_NATIVE = 2242.987061
                COLOR_FY_NATIVE = 2241.741699
                COLOR_CX_NATIVE = 1893.117432
                COLOR_CY_NATIVE = 1091.378052
                COLOR_WIDTH_NATIVE  = 3840
                COLOR_HEIGHT_NATIVE = 2160
                scale_x = width  / COLOR_WIDTH_NATIVE
                scale_y = height / COLOR_HEIGHT_NATIVE
                fx = COLOR_FX_NATIVE * scale_x
                fy = COLOR_FY_NATIVE * scale_y
                cx = COLOR_CX_NATIVE * scale_x
                cy = COLOR_CY_NATIVE * scale_y
                print(f"Effective color resolution: {width}x{height}")
                print(f"Scaled intrinsics -> fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

            # --- MediaPipe ---
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            #last_step = log_step("BGR -> RGB", last_step)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            #last_step = log_step("MediaPipe detection", last_step)

            # --- Depth ---
            if depth_frame.get_format() != OBFormat.Y16:
                continue

            depth_data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape((depth_frame.get_height(), depth_frame.get_width()))

            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
            depth_data = np.where(
                (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH),
                depth_data, 0
            )
            last_step = log_step("Depth processing", last_step)

            current_time = time.time()

            # new_detection = False

            if detection_result.pose_landmarks:
                nose   = detection_result.pose_landmarks[0][0]
                nose_u = int(np.clip(nose.x * width,  0, width  - 1))
                nose_v = int(np.clip(nose.y * height, 0, height - 1))

                nose_z_mm = depth_data[nose_v, nose_u]

                if nose_z_mm > 0:
                    X, Y, Z = deproject_pixel_to_3d(nose_u, nose_v, nose_z_mm, fx, fy, cx, cy)
                    X, Y, Z = convert_to_m(X, Y, Z)
                    #last_step = log_step("3D deprojection", last_step)

                    # --- Rejet des jumps aberrants ---
                    # if last_valid_X is not None:
                    #     dist = ((X - last_valid_X)**2 + (Y - last_valid_Y)**2 + (Z - last_valid_Z)**2) ** 0.5
                        
                    #     print(f"DEBUG: distance={dist:.2f} cm")

                    #     if dist > MAX_JUMP_CM:
                    #         print("DEBUG: JUMP REJETE")
                    #         X, Y, Z = last_valid_X, last_valid_Y, last_valid_Z
                    #     else:
                    #         print("DEBUG: JUMP ACCEPTE")

                    # last_valid_X, last_valid_Y, last_valid_Z = X, Y, Z
                    # new_detection = True

            # --- Si pas de detection valide, on garde la derniere position connue ---
            # if not new_detection:
            #     if last_valid_X is None:
            #         frame_count += 1
            #         key = cv2.waitKey(1)
            #         if key == ord("q") or key == ESC_KEY:
            #             break
            #         continue
            #     X, Y, Z = last_valid_X, last_valid_Y, last_valid_Z

            # --- EMA smoothing ---
                if smooth_X is None:
                    smooth_X, smooth_Y, smooth_Z = X, Y, Z  # init au premier point
                else:
                    smooth_X = EMA_ALPHA * X + (1.0 - EMA_ALPHA) * smooth_X
                    smooth_Y = EMA_ALPHA * Y + (1.0 - EMA_ALPHA) * smooth_Y
                    smooth_Z = EMA_ALPHA * Z + (1.0 - EMA_ALPHA) * smooth_Z
                    #last_step = log_step("EMA smoothing", last_step)


            # if current_time - last_print_time >= PRINT_INTERVAL:
            #     status = "detected" if new_detection else "HOLD last"
            #     print(
            #         f"Nose [{status}] -> "
            #         f"3D X={smooth_X:+.1f}  Y={smooth_Y:+.1f}  Z={smooth_Z:.1f} cm  "
            #         f"-> OSC sent  [{frame_count / max(current_time - fps_timer, 0.001):.1f} fps]"
            #     )
            #     last_print_time = current_time
            #     frame_count = 0
            #     fps_timer   = current_time
                    osc_message = [
                        float(smooth_X),
                        float(smooth_Y),
                        float(smooth_Z)
                    ]
                    osc_client.send_message(OSC_ADDRESS, osc_message)
                    #osc_client.send_message(OSC_ADDRESS, [float(smooth_X), float(smooth_Y), float(smooth_Z)])
                    #last_step = log_step("OSC message sent", last_step)

                    #osc_client.send_message(OSC_ADDRESS, [float(X), float(Y), float(Z)])
                    osc_log.write(
                    f"{get_timestamp_ms()} | "
                    f"OSC -> {OSC_ADDRESS} "
                    f"X={osc_message[0]:+.3f} "
                    f"Y={osc_message[1]:+.3f} "
                    f"Z={osc_message[2]:+.3f}\n"
                )
                label = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"

                cv2.circle(color_image, (nose_u, nose_v), 8, (0, 255, 0), -1)
                cv2.putText(
                    color_image, label,
                    (nose_u + 12, nose_v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                    # if current_time - last_print_time >= PRINT_INTERVAL:
                    #     print(
                    #         f"Nose -> pixel=({nose_u}, {nose_v}) | "
                    #         f"3D X={X:+.1f}  Y={Y:+.1f}  Z={Z:.1f} cm  "
                    #         f"-> OSC sent  [{frame_count / (current_time - fps_timer):.1f} fps]"
                    #     )
                    #     last_print_time = current_time
                    #     frame_count = 0
                    #     fps_timer   = current_time
                
            cv2.imshow("RGB Viewer",   color_image)
            # cv2.imshow("Depth Viewer", depth_visual)
            perf_log.info(
                f"FRAME TOTAL: {(time.perf_counter()-frame_start)*1000:.3f} ms"
            )
            perf_log.info("----------------------")
            frame_count += 1

            key = cv2.waitKey(1)
            if key == ord("q") or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Runtime error:", e)

    stop_event.set()
    acq_thread.join(timeout=2.0)
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()