import importlib
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cv2


@dataclass
class DinoLiteStatus:
    device_id: Optional[str]
    config_hex: str
    config_flags: Dict[str, bool]
    amr: Optional[float]
    fovx_mm: Optional[float]
    stream_width: int
    stream_height: int


class DinoLiteSession:
    """
    One-owner session: opens OpenCV stream + DNX64 SDK once.
    You can query status anytime, and capture an image whenever you're ready.
    """

    def __init__(
        self,
        dnx64_dll_path: str,
        device_index: int = 0,
        cam_index: int = 0,
        desired_size: Tuple[int, int] = (640, 480),
        retries: int = 3,
        sleep_s: float = 0.3,
    ):
        self.dnx64_dll_path = dnx64_dll_path
        self.device_index = device_index
        self.cam_index = cam_index
        self.desired_width, self.desired_height = desired_size
        self.retries = retries
        self.sleep_s = sleep_s

        self._microscope = None
        self._cap = None

    # ---------- lifecycle ----------
    def open(self) -> None:
        """Open OpenCV stream, then initialize the DNX64 SDK."""
        self._cap = self._open_opencv_camera()
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Could not open camera via OpenCV (tried MSMF/DSHOW/DEFAULT)")

        self._microscope = self._open_sdk()

    def close(self) -> None:
        """Release SDK + camera stream."""
        if self._microscope is not None:
            self._close_sdk(self._microscope)
            self._microscope = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # safe even if no windows are open
        cv2.destroyAllWindows()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- querying ----------
    def get_status(self) -> DinoLiteStatus:
        """Return a snapshot of SDK + stream settings."""
        if self._cap is None or self._microscope is None:
            raise RuntimeError("Session not open. Call open() first.")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        device_id = None
        if hasattr(self._microscope, "GetDeviceId"):
            device_id = self._microscope.GetDeviceId(self.device_index)

        config = int(self._microscope.GetConfig(self.device_index))
        flags = self._decode_config_flags(config)

        amr = None
        fovx_mm = None
        if flags.get("AMR", False) and hasattr(self._microscope, "GetAMR"):
            amr = float(self._microscope.GetAMR(self.device_index))
            if hasattr(self._microscope, "FOVx"):
                fov_um = self._microscope.FOVx(self.device_index, amr)
                if fov_um != math.inf:
                    fovx_mm = float(fov_um) / 1000.0

        return DinoLiteStatus(
            device_id=device_id,
            config_hex=f"0x{config:X}",
            config_flags=flags,
            amr=amr,
            fovx_mm=fovx_mm,
            stream_width=w,
            stream_height=h,
        )

    # ---------- streaming + capture ----------
    def read_frame(self) -> Optional[Any]:
        """
        Grab one frame from OpenCV.
        Returns frame (numpy array) or None if frame grab failed.
        """
        if self._cap is None:
            raise RuntimeError("Session not open. Call open() first.")
        ret, frame = self._cap.read()
        return frame if ret else None

    def show_preview(self, window_name: str = "Dino-Lite Preview", delay_ms: int = 1) -> Optional[int]:
        """
        Show current frame. Returns the key code from cv2.waitKey.
        This is optional—useful for visual alignment while moving hardware.
        """
        frame = self.read_frame()
        if frame is None:
            return None
        cv2.imshow(window_name, frame)
        return cv2.waitKey(delay_ms) & 0xFF

    def capture_image(self, filename: Optional[str] = None) -> str:
        """
        Capture and save a single frame to disk.
        If filename not provided, uses timestamp.
        Returns the filename used.
        """
        frame = self.read_frame()
        if frame is None:
            raise RuntimeError("Could not capture image (no frame available).")

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"

        ok = cv2.imwrite(filename, frame)
        if not ok:
            raise RuntimeError(f"Failed to write image to {filename}")

        return filename

    # ---------- internals ----------
    def _open_sdk(self):
        DNX64 = getattr(importlib.import_module("DNX64"), "DNX64")
        microscope = DNX64(self.dnx64_dll_path)

        microscope.SetVideoDeviceIndex(self.device_index)
        time.sleep(0.1)

        for method_name in ("Init", "Initialize", "OpenDevice", "Open"):
            if hasattr(microscope, method_name):
                getattr(microscope, method_name)()
                break

        time.sleep(0.1)
        return microscope

    def _close_sdk(self, microscope):
        for method_name in ("CloseDevice", "Close", "Uninit", "Uninitialize"):
            if hasattr(microscope, method_name):
                getattr(microscope, method_name)()
                break

    def _open_opencv_camera(self):
        backend_candidates = []
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(("MSMF", cv2.CAP_MSMF))
        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(("DSHOW", cv2.CAP_DSHOW))
        backend_candidates.append(("DEFAULT", None))

        for attempt in range(1, self.retries + 1):
            for backend_name, backend in backend_candidates:
                cap = cv2.VideoCapture(self.cam_index) if backend is None else cv2.VideoCapture(self.cam_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # MJPG is usually more stable; doesn't affect your mm/px math
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

                # Request your desired size (no FPS set)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
                time.sleep(0.1)

                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"[OpenCV] backend={backend_name} attempt={attempt}/{self.retries} "
                    f"requested={self.desired_width}x{self.desired_height} got={got_w}x{got_h}"
                )
                return cap

            time.sleep(self.sleep_s)

        return None

    @staticmethod
    def _decode_config_flags(config: int) -> Dict[str, bool]:
        # Mirrors the usb_streamer bit decoding, but returns structured booleans.
        return {
            "EDOF": bool(config & 0x80),
            "AMR": bool(config & 0x40),
            "eFLC": bool(config & 0x20),
            "AimPointLaser": bool(config & 0x10),
            "LED_2seg": (config & 0x0C) == 0x04,
            "LED_3seg": (config & 0x0C) == 0x08,
            "FLC": bool(config & 0x02),
            "AXI": bool(config & 0x01),
        }
