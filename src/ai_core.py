import numpy as np
from collections import deque

# ==========================================
# PART 1: LIVE MONITORING ENGINE (THE CLASS)
# ==========================================
class SurgicalAI:
    def __init__(self):
        # Rolling buffer to store last 10 temperature readings
        self.temp_history = deque(maxlen=10)
        self.last_impedance = 400  # Baseline impedance
        
    # UPDATED FUNCTION SIGNATURE TO ACCEPT 'healthy_tissue_temp'
    def analyze_telemetry(self, current_temp, target_temp, impedance, healthy_tissue_temp=37.0):
        """
        Advanced Heuristic Engine for Thermal Ablation.
        Includes Margin Safety Analysis.
        """
        # 1. Smooth the data
        self.temp_history.append(current_temp)
        if len(self.temp_history) > 0:
            smooth_temp = sum(self.temp_history) / len(self.temp_history)
        else:
            smooth_temp = current_temp
        
        # 2. Calculate Rate of Change (dT/dt)
        if len(self.temp_history) > 1:
            heating_rate = self.temp_history[-1] - self.temp_history[-2] 
        else:
            heating_rate = 0.0

        # 3. Predict Future Temp (Projection 5 seconds ahead)
        predicted_temp_5s = smooth_temp + (heating_rate * 50) 

        # --- SAFETY THRESHOLDS ---
        MAX_SAFE_TEMP = target_temp + 8.0
        CRITICAL_IMPEDANCE_JUMP = 200 

        # ================================
        # LAYER 0: MARGIN SAFETY (NEW)
        # ================================
        # 45°C is where healthy cells start dying (Protein Denaturation)
        if healthy_tissue_temp > 45.0:
            return "STOP", "#FF0000", "⚠️ CRITICAL: HEAT LEAK DETECTED!\nRisk of permanent necrosis in surrounding healthy tissue."
        
        if healthy_tissue_temp > 42.0:
             return "WARNING", "#ff9800", "⚠️ ALERT: Margin approaching unsafe levels.\nCollateral damage risk increasing."

        # LAYER 1: CRITICAL SAFETY
        if current_temp >= MAX_SAFE_TEMP:
            return "STOP", "#FF0000", f"CRITICAL: Temp Limit Exceeded ({current_temp:.1f}°C)!"
        
        if impedance > (self.last_impedance + CRITICAL_IMPEDANCE_JUMP):
            return "STOP", "#FF0000", "EMERGENCY: Impedance Spike! Tissue Charring Detected."

        # LAYER 2: PREDICTIVE SAFETY
        if predicted_temp_5s > MAX_SAFE_TEMP:
            return "PAUSE", "#FFA500", "WARNING: Thermal Runaway Predicted. Pausing to stabilize."

        # LAYER 3: TARGET APPROACH
        remaining = target_temp - smooth_temp
        
        if remaining <= 0:
            if heating_rate > 0.1:
                return "PAUSE", "#FFA500", "Target reached. Cooling down."
            else:
                return "PAUSE", "#28a745", "Target Achieved. Maintaining thermal dose."

        elif remaining < 3.0:
            if heating_rate > 0.5:
                 return "ADJUST", "#FFFF00", "Near Target: Reduce Power."
            else:
                 return "CONTINUE", "#00FF00", "Final Approach. Precision heating active."

        # LAYER 4: NORMAL OPERATION
        else:
            if heating_rate < 0.05 and len(self.temp_history) > 5:
                return "BOOST", "#00d4ff", "Heating inefficient. Suggest increasing power."
            else:
                return "CONTINUE", "#00FF00", f"Stable. Heating at {heating_rate*10:.2f}°C/sec."

    def reset(self):
        self.temp_history.clear()


# ==========================================
# PART 2: PLANNING MODULE (THE FUNCTION)
# ==========================================
def generate_treatment_plan(tumor_size_mm, tumor_type):
    """
    Suggests optimal laser parameters based on tumor biophysics.
    """
    # Defaults
    plan = {
        'power': 10.0,   # Watts
        'energy': 500.0, # Joules
        'target': 60.0,  # Celsius
        'duration': 50.0 # Seconds
    }
    
    tips = []

    # --- LOGIC: TUMOR SIZE ---
    if tumor_size_mm < 15.0:
        plan['power'] = 8.0
        plan['energy'] = 400.0
        tips.append("• Small Lesion: Low power recommended to preserve healthy tissue.")
    elif tumor_size_mm < 30.0:
        plan['power'] = 15.0
        plan['energy'] = 1200.0
        tips.append("• Medium Lesion: Standard ablation protocol applicable.")
    else:
        plan['power'] = 25.0
        plan['energy'] = 2500.0
        tips.append("• Large Mass: High power required. Monitor cooling actively.")

    # --- LOGIC: TUMOR TYPE ---
    if "Glioblastoma" in tumor_type:
        plan['target'] = 65.0 
        tips.append("• Malignancy Detected: Target temp increased to 65°C.")
        tips.append("• Margin Safety: Ablate 2mm beyond visible margin.")
    elif "Benign" in tumor_type:
        plan['target'] = 55.0
        tips.append("• Benign Tissue: Lower temp sufficient.")

    # --- GENERAL PHYSICS ---
    # Calc duration = Energy / Power
    if plan['power'] > 0:
        plan['duration'] = round(plan['energy'] / plan['power'], 1)

    return plan, tips