import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

def segment_eb_roi_v2(vertices, center, refImg, analysis_dir, roi_type, num_roi):
    """
    Divides the EB into ROIs and computes DeltaF/F.
    
    Parameters:
    - rois: List of ROI dictionaries containing 'Vertices' and 'Position'.
    - si_volumes: 3D numpy array representing the imaging volume.
    - analysis_dir: Directory for saving results.
    - roi_type: Method for ROI division ('EqualAngle' or 'EqualArea').
    - num_roi: Number of ROIs.
    
    Returns:
    - sub_rois: List of sub-ROI masks.
    """
    # Extract ROI properties
    temp = np.max(vertices[:,0])-np.min(vertices[:,0])
    temp_two = np.max(vertices[:,1])-np.min(vertices[:,1])
    A = np.min([temp, temp_two])/2  # Semi-major axis
    B = np.max([temp, temp_two])/2  # Semi-minor axis

    # Compute the mean image for visualization
    image_temp = refImg

    # Create elliptical ROI vertices with regular angular spacing
    da = 0.25  # Angular steps in degrees
    angles = np.arange(-270, 90 + da, da)
    ellipse_vertices = []
    ellipse_area = [0]

    for angle in angles:
        pair_vertices = compute_ellipse_vertex(center, angle, A, B)
        ellipse_vertices.append(pair_vertices)
        temp_vertices = np.vstack([center, np.array(ellipse_vertices), center])
        rr, cc = polygon(temp_vertices[:, 1], temp_vertices[:, 0], image_temp.shape)
        mask = np.zeros(image_temp.shape, dtype=bool)
        mask[rr, cc] = True
        ellipse_area.append(mask.sum())

    ellipse_vertices = np.vstack(ellipse_vertices)

    # Create ROIs of either equal angle or equal area
    full_vertices = []
    sub_rois = []
    if roi_type == 'EqualAngle':
        full_angles = np.linspace(angles[0], angles[-1], num_roi + 1)
        for i in range(len(full_angles) - 1):
            angle_start = full_angles[i]
            angle_end = full_angles[i + 1]
            start_idx = np.argmin(np.abs(angles - angle_start))
            end_idx = np.argmin(np.abs(angles - angle_end))
            roi_vertices = np.vstack([center, ellipse_vertices[start_idx:end_idx], center])
            rr, cc = polygon(roi_vertices[:, 1], roi_vertices[:, 0], image_temp.shape)
            mask = np.zeros(image_temp.shape, dtype=bool)
            mask[rr, cc] = True
            sub_rois.append(mask)
            full_vertices.append(roi_vertices)

    # Visualization
    plt.figure()
    plt.imshow(image_temp, cmap='viridis')
    #plt.gca().invert_yaxis()
    for roi in full_vertices:
        plt.plot(roi[:, 0], roi[:, 1], 'r-')
    plt.scatter(center[0], center[1], color='blue', label='Center')
    plt.legend()
    plt.show()

    return sub_rois

def compute_ellipse_vertex(center, angle, a, b):
    """
    Computes the vertex coordinates of an ellipse at a specific angle.
    
    Parameters:
    - center: (x, y) tuple representing the center of the ellipse.
    - angle: Angle in degrees for which the vertex is computed.
    - a: Semi-major axis of the ellipse.
    - b: Semi-minor axis of the ellipse.
    
    Returns:
    - vertices: Coordinates of the ellipse vertex at the given angle.
    """
    angle_rad = np.radians(angle)
    x = center[0] + a * np.cos(angle_rad)
    y = center[1] + b * np.sin(angle_rad)
    return np.array([x, y])


def weighted_circular_mean_time_series(angles, weights):
    """
    Calculate the weighted circular mean for a time series of weights.
    
    Parameters:
        angles (array-like): Angles in radians (1D array, constant for all time steps).
        weights (array-like): Weights for each angle (2D array where rows represent time steps).
    
    Returns:
        numpy.ndarray: Weighted circular mean for each time step (1D array in radians).
    """
    # Ensure inputs are numpy arrays
    angles = np.asarray(angles)
    weights = np.asarray(weights)
    
    # Check dimensions
    if weights.ndim != 2 or weights.shape[1] != len(angles):
        raise ValueError("Weights must be a 2D array with the same number of columns as the length of angles.")
    
    # Weighted sums of sine and cosine for each time step
    S = np.sum(weights * np.sin(angles), axis=1)
    C = np.sum(weights * np.cos(angles), axis=1)
    
    # Circular mean for each time step
    mean_angles = np.arctan2(S, C)
    
    return mean_angles

def correct_circular_discontinuities(signal):
    """
    Corrects discontinuities in a circular signal where it wraps around from pi to -pi or vice versa.
    
    Parameters:
        signal (np.ndarray): A 1D array representing the time-varying circular signal.

    Returns:
        np.ndarray: A 1D array with corrected signal values where discontinuities are replaced by NaN.
    """
    # Compute the difference between consecutive elements
    diff = np.diff(signal)

    # Find where the jump exceeds the discontinuity threshold (e.g., pi in either direction)
    discontinuity_indices = np.where(np.abs(diff) > np.pi)[0]

    # Create a copy of the input signal to avoid modifying the original
    corrected_signal = signal.copy()

    # Replace the values after the discontinuity with NaN
    for idx in discontinuity_indices:
        corrected_signal[idx + 1] = np.nan

    return corrected_signal

def minimize_rms_difference(signal1, signal2):
    """
    Minimize the root mean square difference between two periodic signals
    by rotating (shifting) the second signal.

    Parameters:
        signal1 (numpy array): The reference signal.
        signal2 (numpy array): The target signal to be shifted.

    Returns:
        rotated_signal (numpy array): The rotated target signal.
        optimal_offset (float): The offset value (in radians) applied to the target signal.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Both signals must have the same length.")

    # Discretize offsets for testing
    offsets = np.linspace(-np.pi, np.pi, 1000)  # Test 1000 offsets between -pi and pi
    rms_values = []

    for offset in offsets:
        # Rotate the target signal by the current offset
        rotated_signal = np.mod(signal2 + offset + np.pi, 2 * np.pi) - np.pi
        
        # Compute the RMS difference
        rms = np.sqrt(np.mean((signal1 - rotated_signal) ** 2))
        rms_values.append(rms)

    # Find the offset that minimizes the RMS difference
    optimal_index = np.argmin(rms_values)
    optimal_offset = offsets[optimal_index]

    # Compute the rotated signal using the optimal offset
    rotated_signal = np.mod(signal2 + optimal_offset + np.pi, 2 * np.pi) - np.pi

    return rotated_signal, optimal_offset
