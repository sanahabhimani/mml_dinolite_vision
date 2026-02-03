import argparse
import importlib
import time
import math
import os
import cv2


DNX64_DIR = r"C:\Program Files\DNX64"
os.environ["PATH"] = DNX64_DIR + ";" + os.environ.get("PATH", "")
DNX64_PATH = r"C:\Program Files\DNX64\DNX64.dll"

DEVICE_INDEX = 0     # SDK device index (usually 0)
CAM_INDEX = 0        # OpenCV camera index (I found 0 on my Windows VM)

DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480


QUERY_TIME = 0.05
COMMAND_TIME = 0.25


def load_dnx64():
    DNX64 = getattr(importlib.import_module("DNX64"), "DNX64")
    return DNX64


def init_microscope(microscope):
    microscope.SetVideoDeviceIndex(DEVICE_INDEX)
    time.sleep(0.1)

    for method_name in ("Init", "Initialize", "OpenDevice", "Open"):
        if hasattr(microscope, method_name):
            getattr(microscope, method_name)()
            break

    time.sleep(0.1)
    return microscope


def close_microscope(microscope):
    for method_name in ("CloseDevice", "Close", "Uninit", "Uninitialize"):
        if hasattr(microscope, method_name):
            getattr(microscope, method_name)()
            break


def decode_config_flags(config: int) -> str:
    flags = []
    if (config & 0x80) == 0x80:
        flags.append("EDOF")
    if (config & 0x40) == 0x40:
        flags.append("AMR")
    if (config & 0x20) == 0x20:
        flags.append("eFLC")
    if (config & 0x10) == 0x10:
        flags.append("AimPointLaser")
    if (config & 0xC) == 0x4:
        flags.append("LED_2seg")
    if (config & 0xC) == 0x8:
        flags.append("LED_3seg")
    if (config & 0x2) == 0x2:
        flags.append("FLC")
    if (config & 0x1) == 0x1:
        flags.append("AXI")
    return ", ".join(flags) if flags else "(none)"


def print_fov_mm(microscope):
    amr = microscope.GetAMR(DEVICE_INDEX)
    fov_um = microscope.FOVx(DEVICE_INDEX, amr)

    amr = round(amr, 2)

    if fov_um == math.inf:
        print("FOV: unavailable (AMR out of range)")
    else:
        fov_mm = round(fov_um / 1000.0, 3)
        print(f"FOV: {amr}x -> {fov_mm} mm")


def query_state(microscope):
    if hasattr(microscope, "GetDeviceId"):
        dev_id = microscope.GetDeviceId(DEVICE_INDEX)
        print(f"Device ID: {dev_id}")

    config = microscope.GetConfig(DEVICE_INDEX)
    print(f"Config: 0x{config:X} -> {decode_config_flags(config)}")
    time.sleep(QUERY_TIME)

    if (config & 0x40) == 0x40:
        amr = microscope.GetAMR(DEVICE_INDEX)
        print(f"AMR: {amr}x")
    else:
        print("AMR: flag not set in config (unexpected if your model supports AMR).")

    print_fov_mm(microscope)

def open_opencv_camera(cam_index: int, retries: int = 3, sleep_s: float = 0.3):
    backend_candidates = []
    if hasattr(cv2, "CAP_MSMF"):
        backend_candidates.append(("MSMF", cv2.CAP_MSMF))
    if hasattr(cv2, "CAP_DSHOW"):
        backend_candidates.append(("DSHOW", cv2.CAP_DSHOW))
    backend_candidates.append(("DEFAULT", None))

    for attempt in range(1, retries + 1):
        for backend_name, backend in backend_candidates:
            cap = cv2.VideoCapture(cam_index) if backend is None else cv2.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Force MJPG for better stability (keep this)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

            # Set ONLY the resolution you want (do NOT set FPS)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
            time.sleep(0.1)

            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(
                f"Opened camera using backend: {backend_name} (attempt {attempt}/{retries}), "
                f"requested {DESIRED_WIDTH}x{DESIRED_HEIGHT}, got {got_w}x{got_h}"
            )

            return cap

        time.sleep(sleep_s)

    return None


def print_opencv_stream_info(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"OpenCV stream: {w} x {h} @ {fps:.2f} FPS, FOURCC={fourcc_str}")


def run_query_only():
    DNX64 = load_dnx64()
    microscope = DNX64(DNX64_PATH)

    microscope = init_microscope(microscope)
    query_state(microscope)
    close_microscope(microscope)


def run_preview():
    DNX64 = load_dnx64()
    microscope = DNX64(DNX64_PATH)

    cap = open_opencv_camera(CAM_INDEX, retries=3)
    if cap is None or not cap.isOpened():
        raise RuntimeError("Could not open camera via OpenCV (tried DSHOW/MSMF/DEFAULT)")

    print_opencv_stream_info(cap)

    microscope = init_microscope(microscope)
    print("\n=== SDK state at startup ===")
    query_state(microscope)
    print("Press q or Esc to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Dino-Lite Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    close_microscope(microscope)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dino-Lite SDK + OpenCV tool")
    parser.add_argument(
        "--mode",
        choices=["query", "preview"],
        default="query",
        help="query: print SDK state only. preview: show video + print state once."
    )
    args = parser.parse_args()

    if args.mode == "query":
        run_query_only()
    else:
        run_preview()



'''
# Try a few resolutions until one "sticks"
            for (w, h) in size_candidates:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                time.sleep(0.05)

                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # if the camera accepted (or closely matched) the request, keep it
                if got_w == w and got_h == h:
                    print(f"Opened camera using backend: {backend_name} (attempt {attempt}/{retries}), "
                          f"mode {got_w}x{got_h} MJPG")
                    return cap

            # If none matched exactly, still return the opened cap (better than failing)
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Opened camera using backend: {backend_name} (attempt {attempt}/{retries}), "
                  f"fallback mode {got_w}x{got_h}")
            return cap
'''
