import numpy as np

def measure_tumor_advanced(tumor_mask, pixel_spacing=(0.5, 0.5)):
    """
    Task 3: Advanced Tumor Measurement using PCA.
    Input: Binary Mask (0/255 or 0/1)
    Output: Dictionary of geometric properties
    """
    # Normalize mask to 0 and 1
    binary_mask = (tumor_mask > 127).astype(int)
    
    coords = np.column_stack(np.where(binary_mask == 1))
    
    if coords.shape[0] == 0:
        return None

    sx, sy = pixel_spacing

    # 1. Area
    area_pixels = coords.shape[0]
    area_mm2 = area_pixels * sx * sy

    # 2. Bounding box (Note: numpy returns y,x)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2

    # 3. Diameters
    Dx = (x_max - x_min) * sx
    Dy = (y_max - y_min) * sy

    # 4. Equivalent Diameter
    ECD = 2 * np.sqrt(area_mm2 / np.pi)

    # 5. Shape descriptors using PCA
    try:
        coords_centered = coords - np.array([[y_min, x_min]]) # Align to 0,0 for stability
        cov = np.cov(coords_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        major = 2 * np.sqrt(abs(eigenvalues[0])) # Scale factor approx
        minor = 2 * np.sqrt(abs(eigenvalues[1]))
        
        if major == 0: major = 1 # Avoid div by zero
        if minor == 0: minor = 1

        eccentricity = np.sqrt(1 - (minor**2 / major**2)) if major > minor else 0
        elongation = major / minor
        
    except:
        # Fallback if PCA fails (e.g. single pixel tumor)
        eccentricity = 0.0
        elongation = 1.0

    return {
        "area_mm2": round(area_mm2, 2),
        "center": (int(center_x), int(center_y)), # Tuple (x, y)
        "width_mm": round(Dx, 2),
        "height_mm": round(Dy, 2),
        "equivalent_diameter_mm": round(ECD, 2),
        "eccentricity": round(eccentricity, 3),
        "elongation": round(elongation, 2),
        "bbox": (int(x_min), int(x_max), int(y_min), int(y_max))
    }