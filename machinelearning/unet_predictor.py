from pathlib import Path
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


# utility functions 
def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def normalize_image(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


# unet build function

def build_unet_model(device: str):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(device)
    return model


# model-loading code
def load_unet_weights(model_path: str | Path, device: str):
    model = build_unet_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# prediction function 

def predict_unet_mask(
    image_path: str | Path,
    model,
    device: str,
    threshold: float = 0.5,
):
    image_path = Path(image_path)

    img = load_grayscale(image_path)
    img_norm = normalize_image(img)

    x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask = (prob > threshold).astype(np.uint8)

    return {
        "image_path": str(image_path),
        "raw_image": img,
        "probability_map": prob,
        "binary_mask": pred_mask,
    }


# optional visualization function
def visualize_unet_prediction(
    image_path: str | Path,
    model,
    device: str,
    threshold: float = 0.5,
):
    result = predict_unet_mask(image_path, model, device, threshold=threshold)

    raw_image = result["raw_image"]
    prob = result["probability_map"]
    binary_mask = result["binary_mask"]
    overlay = make_overlay(raw_image, binary_mask)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Raw image")
    plt.imshow(raw_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Pred prob")
    plt.imshow(prob, cmap="magma")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title(f"Pred mask > {threshold}")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# TODO: check that the full frame inference works fine cus right now our patches height and width are 768, 1536. Just check explicitly at NERSC with full reference image

## example usage:
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#unet_model_path = "/global/u2/s/sanbhim/git/mml_dinolite_vision/models/unet_march19_march23_combined.pth"
#model = load_unet_weights(unet_model_path, DEVICE)

#result = predict_unet_mask(
#    image_path="my_new_camera_image.png",
#    model=model,
#    device=DEVICE,
#    threshold=0.22,   # or whatever threshold you found works best
#)

#mask = result["binary_mask"]
#prob_map = result["probability_map"]
#raw_img = result["raw_image"]

### Notes
## raw_image == original grayscale image
## probability_map == continuous Unet output between 0 and 1
## binary_mask == thresholded segmentation mask for downstream use
## for our regressor pipeline, the binary mask is what our regressor and feature-extraction code will use

