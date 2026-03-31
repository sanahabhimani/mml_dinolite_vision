import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class FullImageTouchDeltaRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.scalar_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_img, x_scalar):
        feat_img = self.cnn(x_img).flatten(1)
        feat_scalar = self.scalar_mlp(x_scalar)
        feat = torch.cat([feat_img, feat_scalar], dim=1)
        out = self.head(feat)
        return out


def load_delta_regressor(model_path, device: str):
    model = FullImageTouchDeltaRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def clean_pred_mask(pred_mask, open_len=35, close_len=60):
    bw = (pred_mask > 0).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_len, 5))
    bw_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_len, 3))
    bw_clean = cv2.morphologyEx(bw_open, cv2.MORPH_CLOSE, close_kernel)

    return bw_clean


def find_touch_bands_from_mask(mask, min_run=8, min_gap=20, min_row_sum=50):
    row_sum = (mask > 0).sum(axis=1)

    bands = []
    in_band = False
    start = 0

    for y, s in enumerate(row_sum):
        if (s >= min_row_sum) and (not in_band):
            in_band = True
            start = y
        elif (s < min_row_sum) and in_band:
            end = y - 1
            if (end - start + 1) >= min_run:
                bands.append([start, end])
            in_band = False

    if in_band:
        end = len(row_sum) - 1
        if (end - start + 1) >= min_run:
            bands.append([start, end])

    merged = []
    for b in bands:
        if not merged:
            merged.append(b)
        else:
            if b[0] - merged[-1][1] <= min_gap:
                merged[-1][1] = b[1]
            else:
                merged.append(b)

    return [(b[0], b[1]) for b in merged]


def measure_touches_from_bands(
    mask,
    bands,
    fov_width_mm,
    image_width_px,
    pad_y=5,
    min_width_px=50,
):
    mm_per_px = fov_width_mm / image_width_px
    h, w = mask.shape

    results = []

    for i, (y0, y1) in enumerate(bands):
        ya = max(0, y0 - pad_y)
        yb = min(h, y1 + pad_y)

        roi = (mask[ya:yb, :] > 0).astype(np.uint8)

        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())
        length_px = x2 - x1

        if length_px < min_width_px:
            continue

        y_mid = int((y0 + y1) / 2)

        results.append({
            "touch_id": i,
            "band": (y0, y1),
            "x1": x1,
            "x2": x2,
            "y": y_mid,
            "length_px": float(length_px),
            "length_mm": float(length_px * mm_per_px),
        })

    return results


def build_target_touch_mask(full_mask, touch):
    target_touch_mask = np.zeros_like(full_mask, dtype=np.uint8)

    y0, y1 = touch["band"]
    x1, x2 = touch["x1"], touch["x2"]

    target_touch_mask[y0:y1, x1:x2] = full_mask[y0:y1, x1:x2]
    return target_touch_mask


def build_scalar_features(touch, image_shape, fov_mm):
    h, w = image_shape

    x1 = touch["x1"]
    x2 = touch["x2"]
    y = touch["y"]
    length_px = touch["length_px"]
    seg_length_mm = touch["length_mm"]

    scalar_feats = np.array([
        length_px,
        length_px / w,
        seg_length_mm,
        float(fov_mm),
        x1 / w,
        x2 / w,
        y / h,
    ], dtype=np.float32)

    return scalar_feats


def build_regressor_image_input(raw_image, full_mask, target_touch_mask, out_h=256, out_w=256):
    img = raw_image.astype(np.float32) / 255.0
    full_mask = full_mask.astype(np.float32) / 255.0
    target_touch_mask = target_touch_mask.astype(np.float32) / 255.0

    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    full_mask = cv2.resize(full_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    target_touch_mask = cv2.resize(target_touch_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    x_img = np.stack([img, full_mask, target_touch_mask], axis=0).astype(np.float32)
    return x_img


def predict_touch_delta_mm(
    raw_image,
    full_mask,
    touch,
    fov_mm,
    model,
    device: str,
    image_name=None,
    out_h=256,
    out_w=256,
):
    target_touch_mask = build_target_touch_mask(full_mask, touch)
    x_img = build_regressor_image_input(
        raw_image,
        full_mask,
        target_touch_mask,
        out_h=out_h,
        out_w=out_w,
    )
    x_scalar = build_scalar_features(touch, raw_image.shape, fov_mm)

    x_img_t = torch.from_numpy(x_img).unsqueeze(0).to(device)
    x_scalar_t = torch.from_numpy(x_scalar).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_delta_mm = model(x_img_t, x_scalar_t).item()

    pred_length_mm = touch["length_mm"] + pred_delta_mm

    uid = None
    if image_name is not None:
        uid = f"{image_name}_touch{touch['touch_id']}"

    return {
        "image_name": image_name,
        "uid": uid,
        "touch_id": touch["touch_id"],
        "band": touch["band"],
        "x1": touch["x1"],
        "x2": touch["x2"],
        "y": touch["y"],
        "length_px": touch["length_px"],
        "seg_length_mm": touch["length_mm"],
        "pred_delta_mm": float(pred_delta_mm),
        "pred_length_mm": float(pred_length_mm),
        "target_touch_mask": target_touch_mask,
    }


def predict_all_touches_from_mask(
    raw_image,
    binary_mask,
    fov_mm,
    model,
    device: str,
    image_name=None,
    clean_open_len=35,
    clean_close_len=60,
    band_min_run=8,
    band_min_gap=20,
    band_min_row_sum=50,
    min_width_px=50,
    out_h=256,
    out_w=256,
):
    full_mask = clean_pred_mask(
        binary_mask,
        open_len=clean_open_len,
        close_len=clean_close_len,
    )

    bands = find_touch_bands_from_mask(
        full_mask,
        min_run=band_min_run,
        min_gap=band_min_gap,
        min_row_sum=band_min_row_sum,
    )

    touches = measure_touches_from_bands(
        full_mask,
        bands=bands,
        fov_width_mm=fov_mm,
        image_width_px=raw_image.shape[1],
        min_width_px=min_width_px,
    )

    predictions = []
    for touch in touches:
        pred = predict_touch_delta_mm(
            raw_image=raw_image,
            full_mask=full_mask,
            touch=touch,
            fov_mm=fov_mm,
            model=model,
            device=device,
            image_name=image_name,
            out_h=out_h,
            out_w=out_w,
        )
        predictions.append(pred)

    return {
        "image_name": image_name,
        "raw_image": raw_image,
        "clean_mask": full_mask,
        "bands": bands,
        "touches": touches,
        "predictions": predictions,
    }


def save_prediction_overlay(result, touch_index, save_path):
    raw_image = result["raw_image"]
    clean_mask = result["clean_mask"]
    pred = result["predictions"][touch_index]
    target_touch_mask = pred["target_touch_mask"]

    overlay = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    overlay[clean_mask > 0] = [0, 0, 255]        # blue = all touches
    overlay[target_touch_mask > 0] = [255, 0, 0] # red = selected touch

    cv2.line(
        overlay,
        (int(pred["x1"]), int(pred["y"])),
        (int(pred["x2"]), int(pred["y"])),
        (0, 255, 0),
        3
    )

    label = (
        f"{pred['uid']} | "
        f"seg={pred['seg_length_mm']:.3f} mm | "
        f"final={pred['pred_length_mm']:.3f} mm"
    )

    cv2.putText(
        overlay,
        label,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(save_path, overlay)
    return save_path


def visualize_delta_regressor_result(result, touch_index=0, figsize=(16, 4)):
    raw_image = result["raw_image"]
    clean_mask = result["clean_mask"]
    predictions = result["predictions"]

    if len(predictions) == 0:
        print("No touches found.")
        return

    pred = predictions[touch_index]
    target_touch_mask = pred["target_touch_mask"]

    overlay = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    overlay[clean_mask > 0] = [0, 0, 255]
    overlay[target_touch_mask > 0] = [255, 0, 0]

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.title("Raw image")
    plt.imshow(raw_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Clean mask")
    plt.imshow(clean_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay (target = red)")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"image_name     : {pred['image_name']}")
    print(f"uid            : {pred['uid']}")
    print(f"touch_id       : {pred['touch_id']}")
    print(f"seg_length_mm  : {pred['seg_length_mm']:.3f}")
    print(f"pred_delta_mm  : {pred['pred_delta_mm']:.3f}")
    print(f"pred_length_mm : {pred['pred_length_mm']:.3f}")