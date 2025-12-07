<p align="center">
  <img src=![logo](https://github.com/user-attachments/assets/5c716663-7a42-45a6-8447-c35ca102b67c) alt="OncoSim Logo" width="150"/>
</p>

<h1 align="center">OncoSim: A Technical Overview</h1>
<p align="center">
  <strong>A Numerical Simulation and Guidance Platform for Medical Thermal Ablation</strong>
</p>
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-PyQt5-4E9A06?logo=qt" alt="PyQt5"></a>
  <a href="#"><img src="https://img.shields.io/badge/Libraries-NumPy%2C%20SciPy%2C%20OpenCV-informational" alt="Libraries"></a>
</p>

---

## Abstract

OncoSim is a desktop application developed to model the complex biophysical phenomena involved in laser-induced thermal therapy for oncological applications. The system integrates a multi-stage image processing pipeline for tumor segmentation with a real-time, finite-difference solver for the Pennes Bio-Heat Equation. An external Large Vision Model (LVM) provides high-level semantic analysis, including pathology classification and parameter estimation, which feeds into both the physics simulation and a real-time decision support system. A secondary PDE model based on the Fisher-KPP equation is included to simulate potential tumor growth and invasion dynamics.

![OncoSim Screenshot](screenshot.png)

## Core Architectural Modules & Algorithms

The application's architecture is modular, separating the GUI from the scientific engines. Each module addresses a specific task in the clinical workflow.

### 1. Image Segmentation (`segmentation.py`)

-   **Objective:** Isolate the tumorous region from healthy tissue to generate a binary mask.
-   **Algorithm:** A multi-stage, non-machine learning approach was implemented for robustness and to avoid the need for a large training dataset.
    1.  **Brain Extraction:** A mask of the brain parenchyma is generated via thresholding and morphological erosion to remove the skull and scalp.
    2.  **Anisotropic Diffusion:** The extracted brain region is smoothed using the Perona-Malik equation (via `medpy`) to reduce noise while preserving critical edges.
    3.  **Tumor Proposal:** A hybrid scoring mechanism identifies the most probable tumor region. It combines two metrics:
        -   **Shannon Entropy:** Tumorous regions often exhibit higher textural complexity.
        -   **Intensity Deviation:** Malignant tissues are often hyperintense on T1-contrast or FLAIR sequences.
        A weighted sum of these scores identifies the most "suspicious" cluster from a multi-Otsu thresholding result.
    4.  **Boundary Refinement:** The initial proposal mask is refined using a **PDE-based level-set method** (Multiphase Chan-Vese) to accurately delineate the tumor boundary.
-   **Output:** A binary `(0, 255)` NumPy array representing the tumor mask and a confidence score derived from the suspicion score ratio.

### 2. Geometric Measurement (`tumor_measurement.py`)

-   **Objective:** Quantify the geometric properties of the segmented tumor mask.
-   **Algorithm:** Leverages `skimage.measure.regionprops`, which is more robust than manual moment calculation.
    -   **Centroid:** Calculated as the center of mass of the mask pixels.
    -   **Area & Dimensions:** Extracted from the `area` and `bbox` properties.
    -   **Shape Descriptors:** **Eccentricity** is calculated from the eigenvalues of the region's covariance matrix, providing a measure of elongation (0 for a circle, 1 for a line).
-   **Solved Problem:** Provides precise `(x, y)` coordinates for the laser targeting in the PDE solver and accurate size metrics for physics calculations.

### 3. Cloud AI Analysis (`cloud_ai_engine.py`)

-   **Objective:** Perform high-level radiological analysis that is beyond the scope of classical computer vision or a simple CNN.
-   **Implementation:** The `CombinedAIWorker` sends the segmented scan to a Large Vision Model (Anthropic Claude 3 Haiku via OpenRouter). A structured prompt instructs the model to return a JSON object containing:
    -   `classification` (e.g., "Glioblastoma")
    -   `grade` (e.g., "Grade IV")
    -   `estimated_depth_mm`
    -   `pathology_analysis` (textual description)
    -   `recommendation` (clinical action)
    -   `laser_parameters` (initial suggestion)
-   **Solved Problem:** This offloads the complex tasks of classification and parameter estimation to a state-of-the-art model, removing the need for local trained models (like the VGG19) and providing richer, more detailed diagnostic text. A separate `FollowUpChatWorker` handles conversational context.

### 4. Bio-Heat Simulation (`laser_physics.py`)

-   **Objective:** Simulate the temperature distribution (`T(x, y, t)`) in real-time during laser ablation.
-   **Algorithm:** Solves the **Pennes Bio-Heat Transfer Equation** using a vectorized, explicit finite-difference method.
    
    `ρc * ∂T/∂t = ∇·(k∇T) - ωb*cb*(T - Ta) + Qm + Ql`
    
    -   **Implementation:** The `solve_bioheat_pde` function calculates each term for the entire 2D grid simultaneously using NumPy and OpenCV.
        -   **Diffusion `(∇·(k∇T))`**: The Laplacian `∇²T` is computed efficiently using `cv2.Laplacian`. The model is enhanced to be **temperature-dependent**, where thermal conductivity `k(T)` is a function of the current temperature map `T`.
        -   **Perfusion `(ωb*cb*(T - Ta))`**: The blood perfusion term `ωb(T)` is also dynamic, modeling vasodilation at moderate temperatures and coagulation (shutdown of blood flow) at higher temperatures.
        -   **Laser Source `(Ql)`**: A volumetric heat source is modeled as a 2D Gaussian distribution centered at the tumor centroid.
-   **Solved Problem:** Replaced extremely slow, nested Python `for` loops with a high-performance vectorized solver, enabling a smooth, real-time animation of the heat map in the GUI.

### 5. Tumor Growth Simulation (`tumor_growth_model.py`)

-   **Objective:** Model the potential spatial invasion of the tumor over time.
-   **Algorithm:** Solves the **Fisher-KPP (Kolmogorov-Petrovsky-Piskunov) reaction-diffusion equation**.
    
    `∂u/∂t = D∇²u + ρu(1 - u)`
    
    -   **Implementation:** A vectorized finite-difference scheme is used for performance.
        -   The **diffusion term `(D∇²u)`** (cell migration) is solved with `cv2.Laplacian`.
        -   The **reaction term `(ρu(1 - u))`** (logistic cell growth) is solved with NumPy.
-   **Solved Problem:** Provides a predictive visualization of tumor progression, offering insights beyond static analysis. Replaced an initial, slow sparse-matrix implementation with a faster, image-based approach suitable for a GUI.

## GUI & System Integration (`gui_web_layout.py`)

-   **Framework:** PyQt5.
-   **Architecture:** A multi-threaded approach is used to prevent the GUI from freezing during heavy computations.
    -   `SegmentationWorker`, `CombinedAIWorker`, and `TumorGrowthWorker` offload their respective tasks to background threads (`QThread`).
    -   Results are communicated back to the main GUI thread via Qt's `pyqtSignal` mechanism, ensuring thread safety.
-   **Visualization:**
    -   **Heatmap:** `heatmap_engine.py` uses Matplotlib to render the temperature map from the PDE solver into an image buffer, which is then displayed in a `QLabel`. This allows for advanced features like contour lines and color bars.
    -   **User Interaction:** All user inputs (sliders, text boxes) are dynamically passed to the relevant physics or AI modules.
-   **Reporting:** `report_generator.py` uses the `reportlab` library's Platypus framework to build a multi-page PDF document from a dictionary of all collected session data. It automatically handles text wrapping and page breaks for long chat logs.
