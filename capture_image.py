import time
from dinolite import DinoLiteSession

DNX64_PATH = r"C:\Program Files\DNX64\DNX64.dll"

with DinoLiteSession(
    dnx64_dll_path=DNX64_PATH,
    device_index=0,
    cam_index=0,
    desired_size=(640, 480),   # change if you want
) as s:

    status = s.get_status()
    print(status)

    print("Previewing for alignment... (Ctrl+C to stop preview loop)")
    try:
        for _ in range(300):  # ~10 seconds if delay_ms=30 below
            s.show_preview(delay_ms=30)
    except KeyboardInterrupt:
        pass

    print("Capturing image now...")
    fname = s.capture_image()
    print("Saved:", fname)

    time.sleep(0.2)  # small pause before exit/close
