import numpy as np
from skimage import img_as_float
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes, binary_erosion, label
from medpy.filter.smoothing import anisotropic_diffusion
import cv2

# =============================================================================
#  CORE LOGIC (Extracted from your provided code)
# =============================================================================

def calculate_shannon_entropy(image_region):
    if image_region.size == 0: return 0
    hist = np.histogram(image_region, bins=256, range=(0, 1))[0]
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-9))

def extract_brain_mask(image):
    mean_val = image[image > 0].mean()
    binary_head_mask = image > mean_val * 0.5 
    filled_mask = binary_fill_holes(binary_head_mask)
    struct_el = np.ones((15, 15))
    brain_mask = binary_erosion(filled_mask, structure=struct_el, iterations=2)
    labels, _ = label(brain_mask)
    if labels.max() == 0: return brain_mask
    largest_comp_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    final_brain_mask = labels == largest_comp_label
    shrink_struct = np.ones((10, 10))
    return binary_erosion(final_brain_mask, structure=shrink_struct)

def generate_tumor_proposal_with_hybrid_score(image, brain_mask):
    brain_pixels = image[brain_mask]
    if brain_pixels.size < 100: return None, 0, 0
    try:
        thresholds = threshold_multiotsu(brain_pixels, classes=4)
        region_map = np.digitize(image, bins=thresholds)
    except:
        return None, 0, 0 # Fallback if Otsu fails
        
    region_map[~brain_mask] = 0
    mean_brain_intensity = brain_pixels.mean()
    std_brain_intensity = brain_pixels.std()
    if std_brain_intensity == 0: std_brain_intensity = 1e-9

    cluster_data = []
    for i in range(1, region_map.max() + 1):
        cluster_mask = region_map == i
        cluster_pixels = image[cluster_mask]
        if cluster_pixels.size == 0: continue
        entropy = calculate_shannon_entropy(cluster_pixels)
        mean_intensity = cluster_pixels.mean()
        cluster_data.append({'label': i, 'entropy': entropy, 'intensity': mean_intensity})

    if not cluster_data: return None, 0, 0

    max_entropy_found = max(d['entropy'] for d in cluster_data)
    for data in cluster_data:
        intensity_score = abs(data['intensity'] - mean_brain_intensity) / std_brain_intensity
        entropy_score = data['entropy'] / max_entropy_found if max_entropy_found > 0 else 0
        data['suspicion_score'] = (1.5 * intensity_score) + (0.5 * entropy_score)

    best_candidate = max(cluster_data, key=lambda x: x['suspicion_score'])
    
    max_suspicion_score = best_candidate['suspicion_score']
    other_scores = [d['suspicion_score'] for d in cluster_data if d['label'] != best_candidate['label']]
    mean_other_scores = np.mean(other_scores) if other_scores else 0

    initial_mask = (region_map == best_candidate['label'])
    labels, num_features = label(initial_mask)
    if num_features == 0: return None, max_suspicion_score, mean_other_scores
    
    largest_object_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    proposal_mask = labels == largest_object_label
    return proposal_mask, max_suspicion_score, mean_other_scores

def refine_with_multiphase_pde(image, proposal_phi, brain_phi, max_iter=50, dt=0.05, mu1=0.1, mu2=0.2):
    phi1, phi2 = proposal_phi.copy(), brain_phi.copy()
    epsilon = 1.0
    for i in range(max_iter):
        H_phi1=0.5*(1+(2/np.pi)*np.arctan(phi1/epsilon)); H_phi2=0.5*(1+(2/np.pi)*np.arctan(phi2/epsilon))
        denom11=np.sum(H_phi1*H_phi2)+1e-8; denom10=np.sum(H_phi1*(1-H_phi2))+1e-8
        denom01=np.sum((1-H_phi1)*H_phi2)+1e-8; denom00=np.sum((1-H_phi1)*(1-H_phi2))+1e-8
        c11=np.sum(image*H_phi1*H_phi2)/denom11; c10=np.sum(image*H_phi1*(1-H_phi2))/denom10
        c01=np.sum(image*(1-H_phi1)*H_phi2)/denom01; c00=np.sum(image*(1-H_phi1)*(1-H_phi2))/denom00
        force1 = (-(image-c11)**2 + (image-c01)**2)*H_phi2 + (-(image-c10)**2 + (image-c00)**2)*(1-H_phi2)
        dirac_phi1 = (epsilon/np.pi)/(epsilon**2+phi1**2)
        phi1_y, phi1_x = np.gradient(phi1); grad_phi1_norm = np.sqrt(phi1_x**2+phi1_y**2+1e-8)
        div_nx1_y, _ = np.gradient(phi1_x/grad_phi1_norm); _, div_ny1_x = np.gradient(phi1_y/grad_phi1_norm)
        curvature1 = div_nx1_y+div_ny1_x
        dphi1_dt = dirac_phi1*(mu1*curvature1+force1)
        phi1 += dt*dphi1_dt
    return phi1

# =============================================================================
#  INTERFACE FUNCTION (Called by GUI)
# =============================================================================

def detect_tumor(image_bgr):
    """
    Input: OpenCV Image (BGR, 0-255)
    Output: Dictionary with mask and confidence, plus brain_mask (as uint8)
    """
    try:
        # 1. Preprocess: Convert to Float Grayscale (0.0 - 1.0)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        img_float = img_as_float(gray)
        
        # 2. Extract Brain (We need to return this)
        brain_mask = extract_brain_mask(img_float)
        brain_mask_uint8 = (brain_mask * 255).astype(np.uint8)

        if not np.any(brain_mask): 
            return {"found": False, "brain_mask": None}

        # 3. Smoothing
        brain_only = img_float * brain_mask
        processed = anisotropic_diffusion(brain_only, niter=15, kappa=50, gamma=0.1)
        
        # 4. Generate Proposal
        proposal_mask, max_score, mean_score = generate_tumor_proposal_with_hybrid_score(processed, brain_mask)
        
        # 5. Check Confidence
        if mean_score == 0: 
            ratio = 100
        else: 
            ratio = max_score / mean_score
        
        confidence = 0.0
        if ratio > 5.0: 
            confidence = 99.0
        elif ratio > 2.0: 
            confidence = 50.0 + ((ratio-2.0)/3.0)*50.0
        
        # 6. Logic Gate: Is it a tumor?
        if ratio < 1.5 or proposal_mask is None:
            return {
                "found": False,
                "mask": np.zeros(gray.shape, dtype=np.uint8),
                "confidence": confidence,
                "scores": (max_score, mean_score)
            }
            
        # 7. Refine with PDE
        proposal_phi = np.where(proposal_mask, -2.0, 2.0)
        brain_phi = np.where(brain_mask, -2.0, 2.0)
        final_phi = refine_with_multiphase_pde(processed, proposal_phi, brain_phi)
        
        refined_mask = binary_fill_holes(final_phi < 0)
        
        # Convert boolean mask back to uint8 (0, 255) for OpenCV
        final_mask_uint8 = (refined_mask * 255).astype(np.uint8)
        
        return {
            "found": True,
            "mask": final_mask_uint8,
            "confidence": confidence,
            "scores": (max_score, mean_score),
            "brain_mask": brain_mask_uint8
        }

    except Exception as e:
        print(f"Advanced Segmentation Failed: {e}")
        return {"found": False, "brain_mask": None}