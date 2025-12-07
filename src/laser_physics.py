import math
import random

# ==========================================
# 1. PHYSICS CALCULATION (Task 4 with Wavelength Logic)
# ==========================================
def calculate_laser_params(size_mm, depth_mm, wavelength_nm):
    """
    Calculates laser parameters using wavelength-specific absorption coefficients.
    
    Args:
        size_mm (float): Tumor diameter in millimeters.
        depth_mm (float): Depth from surface in millimeters.
        wavelength_nm (int): Laser wavelength in nanometers (e.g., 980, 1064).
    """
    # --- 1. WAVELENGTH-SPECIFIC ABSORPTION (µa) in BRAIN TISSUE ---
    # This data is from biomedical optics literature. It's a simplified
    # lookup table for common medical laser wavelengths.
    
    # Key: Wavelength (nm), Value: Absorption Coefficient (1/mm)
    brain_absorption = {
        810:  0.08,  # Diode (Deep Penetration)
        980:  0.15,  # Diode (Balanced)
        1064: 0.12,  # Nd:YAG (Deep Penetration)
        1320: 0.35,  # Nd:YAG (More superficial)
        1470: 1.20,  # Diode (High water absorption, very superficial)
        10600: 20.0  # CO2 (Extremely high water absorption, surface only)
    }
    
    # Find the closest matching coefficient from our known data
    closest_wavelength = min(brain_absorption.keys(), key=lambda k: abs(k - wavelength_nm))
    mu_a = brain_absorption[closest_wavelength]
    
    print(f"Physics: Using µa={mu_a} for wavelength ~{closest_wavelength}nm")

    # --- 2. TEMPERATURE & PENETRATION (Beer-Lambert Law) ---
    target_temp = 65.0
    penetration_factor = math.exp(-mu_a * depth_mm)
    required_intensity = 1.0 / max(penetration_factor, 1e-6) # Avoid division by zero

    # --- 3. ENERGY & POWER (Q = m*c*ΔT) ---
    radius = size_mm / 2
    tumor_volume_mm3 = (4/3) * math.pi * (radius**3)
    tumor_volume_cm3 = tumor_volume_mm3 / 1000  

    temp_increase = target_temp - 37.0
    energy = tumor_volume_cm3 * 4.18 * temp_increase
    energy *= required_intensity # Adjust for depth

    # Standard 30-second procedure target
    duration = 30.0
    power = energy / duration

    # Safety clamp (max 100W for medical lasers)
    if power > 100: 
        power = 100
        duration = energy / power

    return {
        "energy_J": round(energy, 2),
        "power_W": round(power, 2),
        "duration_s": round(duration, 1),
        "target_temperature_C": target_temp
    }


# ==========================================
# 2. PDE SIMULATION (For Heatmap Animation)
# ==========================================
# This part remains the same as it simulates the visual effect of the
# calculated power, rather than re-calculating it.
def calculate_pde_state(current_temp, target_temp, power, mask_area):
    """
    Simulates heat diffusion focusing on the Centroid (Laser Tip).
    
    Returns:
    - next_temp: Temperature at the CENTROID (Hottest point).
    - margin_temp: Temperature at the TUMOR EDGE (Cooler due to Gaussian decay).
    - is_destroyed: If the Centroid has reached the target.
    """
    # --- 1. PHYSICAL CONSTANTS ---
    rho = 1050.0       # Density (kg/m^3)
    c_p = 3600.0       # Specific Heat (J/kg*K)
    k_cond = 0.52      # Conductivity (W/m*K)
    dt = 0.1           # Time step

    # --- 2. FOCUSING THE LASER ( The "Centroid" Logic ) ---
    # The laser doesn't heat the whole tumor instantly. It heats the tip.
    # We define an "Optical Penetration Volume" (small area around the tip).
    # This ensures the center gets HOT quickly, while the rest waits for conduction.

    # Even if tumor is huge (mask_area is big), the laser tip is small.
    # We use a weighted volume: mostly the tip, slightly affected by total size.
    tip_volume_m3 = (15 + mask_area * 0.05) * (1e-6) * 0.005 

    mass_at_tip = rho * tip_volume_m3

    # --- 3. HEAT SOURCE (At the Center) ---
    # Energy injected directly into the centroid
    energy_in = float(power) * dt
    temp_rise = energy_in / (mass_at_tip * c_p)

    # --- 4. COOLING (Perfusion & Conduction) ---
    delta_T = current_temp - 37.0

    # Dynamic Blood Flow (Vasodilation -> Shutdown)
    base_perfusion = 0.004 
    if current_temp > 60.0:
        w_b = 0.0  # Vascular shutdown (Coagulation)
    elif current_temp > 37.0:
        # Blood vessels widen to cool the hot center
        w_b = base_perfusion * (1 + 3.5 * (1 - math.exp(-0.5 * delta_T)))
    else:
        w_b = base_perfusion

    # Perfusion cooling at the center
    cooling_perfusion = (w_b * 3800 * 1060 * delta_T * dt) / (rho * c_p)

    # Conduction cooling (Heat moving from Center -> Outwards)
    # As the center gets hotter, it loses more heat to the edges
    cooling_conduction = (k_cond * delta_T * dt) / (rho * c_p * 0.005)

    # --- 5. CALCULATE CORE TEMP (Centroid) ---
    next_temp = current_temp + temp_rise - cooling_perfusion - cooling_conduction

    # --- 6. CALCULATE MARGIN TEMP (Gaussian Decay) ---
    # The edge is cooler because heat follows a Bell Curve (Gaussian).
    # Typically, if Center is Peak, the edge is at ~1/e (~37%) or 1/e^2 (~13%) depending on radius.
    # We simulate this spatial drop-off:
    
    # As the tumor gets bigger (mask_area), the edge is further away, so it's cooler.
    # Small tumor = Edge is close to center = Hotter edge.
    # Large tumor = Edge is far from center = Cooler edge.
    dist_factor = 1.0 / (1.0 + math.sqrt(mask_area) * 0.1)

    core_rise = next_temp - 37.0
    margin_rise = core_rise * 0.45 * dist_factor # 45% of core temp, reduced by distance

    margin_temp = 37.0 + margin_rise

    # --- 7. CHECK STATUS ---
    # Destruction happens if the *Margin* reaches lethal temp (total ablation)
    # OR if we just want to track the core safety.
    is_destroyed = next_temp >= target_temp

    return next_temp, margin_temp, is_destroyed