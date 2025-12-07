import numpy as np
import cv2
import time

def simulate_tumor_growth_fast(initial_mask: np.ndarray, brain_mask: np.ndarray, params: dict, end_time: int, save_every: int = 5):
    """
    Fast, vectorized Fisher-KPP simulation.
    - Constrained by the brain mask.
    - Correctly calculates steps based on the desired end_time.
    """
    # 1. Extract parameters
    D = params.get('D', 0.8)
    rho = params.get('rho', 0.5)
    beta = params.get('beta', 0.1)
    time_scale = params.get('time_scale', 'hours')
    pixel_scale_mm = 0.5

    # --- FIX: Calculate number of steps based on end_time ---
    if time_scale == 'days':
        # If one step is 2.4 hours, how many steps to reach end_time days?
        dt = 2.4
        steps = int((end_time * 24) / dt)
    else:  # hours
        # If one step is 0.1 hours, how many steps to reach end_time hours?
        dt = 0.1
        steps = int(end_time / dt)
    # --------------------------------------------------------

    # 2. Initialize
    u = (initial_mask.astype(np.float32) / 255.0).clip(0, 1)
    brain_mask_float = (brain_mask.astype(np.float32) / 255.0).clip(0, 1)

    initial_pixels = np.sum(u > 0.5)
    initial_radius_mm = np.sqrt(initial_pixels * (pixel_scale_mm**2) / np.pi) if initial_pixels > 0 else 0

    frames = []
    metrics = []

    # 3. Simulation Loop
    for step in range(1, steps + 1):
        if step % 20 == 0:
            time.sleep(0.01)

        laplacian = cv2.Laplacian(u, cv2.CV_32F)
        reaction = rho * u * (1 - u) - beta * u
        change_in_u = dt * (D * laplacian + reaction)

        # Apply brain mask to stop growth
        u += (change_in_u * brain_mask_float)

        u = np.clip(u, 0.0, 1.0)

        if step % save_every == 0 or step == steps:
            # Calculate metrics for this frame
            current_pixels = np.sum(u > 0.5)
            current_radius_mm = np.sqrt(current_pixels * (pixel_scale_mm**2) / np.pi) if current_pixels > 0 else 0
            growth_delta_mm = current_radius_mm - initial_radius_mm

            core_density = np.mean(u[u > 0.8]) if np.any(u > 0.8) else 0
            grade_change = "Increased" if core_density > 0.95 and rho > 0.5 else "Stable"

            # --- FIX: Calculate current time for metrics ---
            current_time = step * dt
            if time_scale == 'days':
                current_time /= 24 # Convert hours back to days
            # -----------------------------------------------

            frame_metrics = {
                "current_time": current_time,
                "growth_delta_mm": growth_delta_mm,
                "grade_status": grade_change
            }
            metrics.append(frame_metrics)

            # Save visual frame
            frame_img = (u * 255).astype(np.uint8)
            frames.append(cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR))

    return frames, metrics