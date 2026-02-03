import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def preprocess_image_blackhat(
    image_path,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    blackhat_ksize=51,
    blur_ksize=5,
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blackhat_ksize, blackhat_ksize))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    blurred = cv2.GaussianBlur(blackhat, (blur_ksize, blur_ksize), 0)

    return img, blackhat, blurred


def extract_horizontal_mask(blurred_blackhat, binarize="otsu", open_len=250, close_len=120):
    """
    Turn blackhat into a binary mask of long horizontal features (scratches).
    """
    # --- Binarize ---
    if binarize == "otsu":
        # Otsu chooses threshold automatically
        _, bw = cv2.threshold(blurred_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # fallback fixed threshold
        _, bw = cv2.threshold(blurred_blackhat, 30, 255, cv2.THRESH_BINARY)

    # --- Morphology to keep long horizontals and kill speckle ---
    # Opening removes small junk but preserves long horizontal structures
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_len, 3))
    bw_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel)

    # Closing reconnects broken segments along the horizontal direction
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_len, 3))
    bw_clean = cv2.morphologyEx(bw_open, cv2.MORPH_CLOSE, close_kernel)

    return bw, bw_clean


def find_cut_bands_from_mask(mask, min_run=12, min_gap=18, min_row_sum=200):
    """
    Find horizontal bands (y-ranges) where the mask has lots of white pixels.
    Returns list of (y0, y1) inclusive bounds for each band.
    """
    # mask is 0/255; convert to 0/1 then sum across x
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

    # close if ended inside a band
    if in_band:
        end = len(row_sum) - 1
        if (end - start + 1) >= min_run:
            bands.append([start, end])

    # merge bands separated by small gaps (multiple ridges within one physical cut)
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


def merge_bands_by_x_overlap(mask, bands, gap_tol=80, overlap_frac=0.6):
    """
    Merge adjacent bands if they are close in y AND their x support overlaps strongly.
    """
    if not bands:
        return []

    def x_extent(y0, y1):
        band = mask[y0:y1+1, :]
        ys, xs = np.where(band > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(xs.max())

    merged = [list(bands[0])]
    for (y0, y1) in bands[1:]:
        prev_y0, prev_y1 = merged[-1]
        if y0 - prev_y1 > gap_tol:
            merged.append([y0, y1])
            continue

        prev_ext = x_extent(prev_y0, prev_y1)
        cur_ext  = x_extent(y0, y1)
        if (prev_ext is None) or (cur_ext is None):
            merged.append([y0, y1])
            continue

        px1, px2 = prev_ext
        cx1, cx2 = cur_ext

        # overlap length / smaller length
        overlap = max(0, min(px2, cx2) - max(px1, cx1))
        denom = max(1, min(px2 - px1, cx2 - cx1))
        frac = overlap / denom

        if frac >= overlap_frac:
            merged[-1][1] = y1   # merge
        else:
            merged.append([y0, y1])

    return [(a, b) for a, b in merged]


def measure_cuts_from_bands_mask_energy(gate, bands, *,
                                       fov_width_mm, image_width_px,
                                       pad_y=8, min_width_px=800,
                                       energy_frac=0.995,   # higher is fine for binary
                                       smooth_px=31,
                                       gap_fill_px=160):
    """
    Measure x-extent from a binary gate mask using cumulative 'area' energy.
    Much more stable than using blackhat intensity when background texture is strong.
    """
    mm_per_px = fov_width_mm / image_width_px
    cuts = []

    gate01 = (gate > 0).astype(np.uint8)
    H, W = gate01.shape

    k = max(3, smooth_px | 1)
    kernel = np.ones(k, dtype=np.float32) / k

    for (y0, y1) in bands:
        ya = max(0, y0 - pad_y)
        yb = min(H, y1 + pad_y)

        roi = gate01[ya:yb, :]

        # x-profile = how many white pixels in each column
        prof = roi.sum(axis=0).astype(np.float32)

        # smooth
        prof_s = np.convolve(prof, kernel, mode="same")

        total = prof_s.sum()
        if total <= 0:
            continue

        # Fill small gaps in x support
        support = (prof_s > 0).astype(np.uint8)
        if gap_fill_px and gap_fill_px > 1:
            support = cv2.dilate(
                support.reshape(1, -1),
                cv2.getStructuringElement(cv2.MORPH_RECT, (gap_fill_px, 1)),
                iterations=1
            ).ravel()
            prof_s = prof_s * support

        cdf = np.cumsum(prof_s)
        left_idx  = int(np.searchsorted(cdf, (1 - energy_frac) * total))
        right_idx = int(np.searchsorted(cdf, energy_frac * total))

        x1 = max(0, min(W - 1, left_idx))
        x2 = max(0, min(W - 1, right_idx))
        length_px = float(max(0, x2 - x1))

        if length_px < min_width_px:
            continue

        y_mid = int((y0 + y1) / 2)
        cuts.append({
            "x1": x1, "y1": y_mid,
            "x2": x2, "y2": y_mid,
            "length_px": length_px,
            "length_mm": length_px * mm_per_px,
            "band": (y0, y1),
        })

    return cuts


def visualize_bands_and_cuts(image_path, bw_clean, cuts):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()

    for c in cuts:
        y0, y1 = c["band"]
        # band box
        cv2.rectangle(overlay, (0, y0), (rgb.shape[1]-1, y1), (0, 0, 255), 2)
        # measured cut line
        cv2.line(overlay, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (0, 255, 0), 2)

    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1); plt.imshow(bw_clean, cmap="gray"); plt.title("Binary (horizontal-cleaned)"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(overlay); plt.title(f"Cuts: {len(cuts)}"); plt.axis("off")
    plt.tight_layout()
    plt.show()


### Example Usage ###
'''
image_path = "/Users/sanuma/sanahbhim/MLnotebooks/DinoLiteImages/test5_76x_EDR.jpg"

orig, blackhat, blurred = preprocess_image_blackhat(
    image_path,
    clip_limit=2.0,
    blackhat_ksize=51,
    blur_ksize=5
)

bw_raw, bw_clean = extract_horizontal_mask(
    blurred,
    binarize="otsu",
    open_len=250,
    close_len=120
)

# Gate: widen in X, but KEEP Y VERY TIGHT
bw_gate = cv2.dilate(
    (bw_clean > 0).astype(np.uint8) * 255,
    cv2.getStructuringElement(cv2.MORPH_RECT, (241, 3)),  # <-- changed (X bigger, Y smaller)
    iterations=1
)

bands = find_cut_bands_from_mask(
    bw_clean,
    min_run=10,
    min_gap=80,
    min_row_sum=150
)

bands_merged = merge_bands_by_x_overlap(
    bw_clean,
    bands,
    gap_tol=80,
    overlap_frac=0.6
)

cuts = measure_cuts_from_bands_mask_energy(
    bw_gate,
    bands_merged,
    fov_width_mm=4.974,
    image_width_px=2592,
    pad_y=8,
    min_width_px=800,
    energy_frac=0.995,   # binary -> use higher
    smooth_px=31,
    gap_fill_px=260
)

for i, c in enumerate(cuts):
    print(f"Cut {i}: {c['length_px']:.1f} px  →  {c['length_mm']:.3f} mm")

visualize_bands_and_cuts(image_path, bw_clean, cuts)

'''
