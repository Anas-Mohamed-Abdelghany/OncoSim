import cv2
import numpy as np

def generate_heatmap(base_image, current_temp, target_temp, baseline_temp=37.0, mask=None):
    """
    Generates thermal overlay using a dynamic baseline temperature.
    """
    if base_image is None:
        return None

    h, w = base_image.shape[:2]

    # --- FIX IS HERE: Use dynamic baseline_temp ---
    delta = current_temp - baseline_temp
    rnge = (target_temp + 10.0) - baseline_temp

    # Fix: Correct handling for numpy arrays to avoid ValueError
    # If rnge is an array, use .all(); otherwise compare as usual
    if np.isscalar(rnge):
        rnge_is_nonpositive = rnge <= 0
    else:
        rnge_is_nonpositive = np.all(rnge <= 0)

    if rnge_is_nonpositive:
        intensity = 1.0
    else:
        intensity = np.clip(delta / rnge, 0, 1.0)
    # ---------------------------------------------

    if np.isscalar(intensity):
        intensity_too_low = intensity <= 0.05
    else:
        intensity_too_low = np.all(intensity <= 0.05)
    if intensity_too_low:
        return base_image

    heatmap_layer = np.zeros((h, w), dtype=np.uint8)

    if mask is not None:
        # If intensity is a scalar or array, broadcast appropriately
        value = int(255 * intensity) if np.isscalar(intensity) else (255 * intensity).astype(np.uint8)
        heatmap_layer[mask == 255] = value
        heatmap_layer = cv2.GaussianBlur(heatmap_layer, (21, 21), 0)
    else:
        cv2.circle(heatmap_layer, (w // 2, h // 2), 50, int(255 * intensity), -1)

    heatmap_color = cv2.applyColorMap(heatmap_layer, cv2.COLORMAP_JET)

    mask_indices = heatmap_layer > 10

    result = base_image.copy()
    result[mask_indices] = cv2.addWeighted(base_image, 0.6, heatmap_color, 0.4, 0)[mask_indices]

    return result