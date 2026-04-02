## how to use the dinolite vision code within automation1 
## assuming imports of DNX64 work fine etc etc:

from pathlib import Path
from dinolite import DinoLiteSession, build_testtouch_image_name

output_dir = Path(r"C:\Users\University of Chicag\git\April2_TrueTestTouches_Thin")
image_name = build_testtouch_image_name(
    "ccat350", "surface1", "90deg", "spindleB", "thick", 1
)
output_path = output_dir / image_name

with DinoLiteSession(
    dnx64_dll_path=r"C:\Program Files\DNX64\DNX64.dll",
    device_index=0,
    cam_index=0,
    desired_size=(2048, 1536),
) as s:
#    print("status:", s.get_status())
    result = s.capture_image(output_path)

print("result:", result)
