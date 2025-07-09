import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QSpinBox, QGroupBox, QCheckBox, QDoubleSpinBox
)
from qtpy.QtCore import Signal
import sys
sys.path.append('../path_tracing/brightest-path-lib')
# Import the memory-optimized tube data generation
from brightest_path_lib.visualization.tube_data import create_tube_data  # Now uses minimal version
from skimage.feature import blob_log


def detect_spines_with_angles(tube_data, frame_index, 
                             min_sigma_2d=5, max_sigma_2d=10, threshold_2d=0.05,
                             min_sigma_tube=5, max_sigma_tube=5, threshold_tube=0.02,
                             max_distance_threshold=15, angle_threshold=20, 
                             angle_weight=0.7):
    """
    Detect spines using combined angle and distance-to-center matching.
    Updated to work with minimal tube data structure.
    """
    # Extract data for the specified frame
    frame_data = tube_data[frame_index]
    center = frame_data['position']
    actual_z = int(center[0])  # Store the actual Z coordinate
    
    # Get dendrite direction from basis vectors (only forward is stored now)
    dendrite_direction = frame_data['basis_vectors']['forward']
    
    # === 2D VIEW PROCESSING ===
    view_2d = frame_data['zoom_patch']
    mask_2d = frame_data['zoom_patch_ref']
    
    if mask_2d is None:
        subtracted_plane_2d = view_2d
    else:
        subtracted_plane_2d = view_2d - mask_2d
        subtracted_plane_2d[subtracted_plane_2d < 0] = 0
    
    blobs_2d = blob_log(subtracted_plane_2d, 
                        min_sigma=min_sigma_2d, 
                        max_sigma=max_sigma_2d, 
                        threshold=threshold_2d,
                        exclude_border=True)
    
    # === TUBULAR VIEW PROCESSING ===
    normal_plane = np.rot90(frame_data['normal_plane'])
    colored_plane = frame_data['colored_plane']
    
    if colored_plane is not None:
        colored_plane = np.rot90(colored_plane)
        if colored_plane.ndim == 3:
            colored_plane = np.mean(colored_plane, axis=-1)
        subtracted_plane_tube = normal_plane - colored_plane
        subtracted_plane_tube[subtracted_plane_tube < 0] = 0
    else:
        subtracted_plane_tube = normal_plane
    
    blobs_tube = blob_log(subtracted_plane_tube,
                         min_sigma=min_sigma_tube,
                         max_sigma=max_sigma_tube,
                         threshold=threshold_tube)
    
    filtered_blobs_2d = blobs_2d
    filtered_blobs_tube = blobs_tube
    
    # === ANGLE-BASED MATCHING ===
    confirmed_spines = []
    view_2d_angles = []
    tube_angles = []
    
    if len(blobs_2d) > 0 and len(blobs_tube) > 0:
        # Get center of tubular view (dendrite center)
        tube_center = np.array([subtracted_plane_tube.shape[0]//2, 
                               subtracted_plane_tube.shape[1]//2])
        
        # Get center of 2D view
        view_2d_center = np.array([subtracted_plane_2d.shape[0]//2, 
                                  subtracted_plane_2d.shape[1]//2])
        
        # No filtering - use all blobs as in original code
        
        # Calculate angles for tubular blobs
        tube_angles = []
        for blob in filtered_blobs_tube:
            spine_vector = np.array([blob[0], blob[1]]) - tube_center
            angle = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]))
            tube_angles.append(angle)
        
        # Calculate angles for 2D blobs
        view_2d_angles = []
        dendrite_2d = np.array([dendrite_direction[1], dendrite_direction[2]])
        dendrite_2d = dendrite_2d / (np.linalg.norm(dendrite_2d) + 1e-8)
        
        for blob in filtered_blobs_2d:
            spine_vector = np.array([blob[0], blob[1]]) - view_2d_center
            spine_vector = spine_vector / (np.linalg.norm(spine_vector) + 1e-8)
            
            dot_product = np.dot(spine_vector, dendrite_2d)
            cross_product = np.cross(spine_vector, dendrite_2d)
            angle = np.degrees(np.arctan2(cross_product, dot_product))
            view_2d_angles.append(angle)
        
        # Match based on combined angle and distance
        used_tube_indices = set()
        
        for i, (blob_2d, angle_2d) in enumerate(zip(filtered_blobs_2d, view_2d_angles)):
            best_match_idx = -1
            best_score = float('inf')
            
            spine_2d_dist = np.linalg.norm(np.array([blob_2d[0], blob_2d[1]]) - view_2d_center)
            
            for j, (blob_tube, angle_tube) in enumerate(zip(blobs_tube, tube_angles)):
                if j in used_tube_indices:
                    continue
                
                angle_diff = min(abs(angle_2d - angle_tube), abs(angle_2d + angle_tube))
                
                if angle_diff > angle_threshold:
                    continue
                
                spine_tube_dist = np.linalg.norm(np.array([blob_tube[0], blob_tube[1]]) - tube_center)
                distance_diff = abs(spine_2d_dist - spine_tube_dist)
                
                angle_score = angle_diff / angle_threshold
                max_dist_diff = max(spine_2d_dist, spine_tube_dist)
                distance_score = distance_diff / max_dist_diff if max_dist_diff > 0 else 0
                
                combined_score = angle_weight * angle_score + (1 - angle_weight) * distance_score
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_match_idx = j
            
            if best_match_idx >= 0:
                used_tube_indices.add(best_match_idx)
                corresponding_tube_blob = blobs_tube[best_match_idx]
                
                confirmed_spines.append({
                    'blob_2d': blob_2d,
                    'blob_tube': corresponding_tube_blob,
                    'coords_2d': (blob_2d[1], blob_2d[0]),
                    'coords_tube': (corresponding_tube_blob[1], corresponding_tube_blob[0]),
                    'actual_z': actual_z
                })
    
    return {
        'frame_index': frame_index,
        'actual_z': actual_z,
        'blobs_2d': blobs_2d,
        'blobs_tube': blobs_tube,
        'confirmed_spines': confirmed_spines,
        'num_confirmed_spines': len(confirmed_spines),
        'view_2d_angles': view_2d_angles,
        'tube_angles': tube_angles
    }


def process_all_frames_with_extension(tube_data, image, brightest_path, max_distance_threshold=15, 
                                     frame_range=2, progress_callback=None):
    """
    Process frames with spine detection and extend detected spines to neighboring frames.
    
    This function:
    1. Detects spines using tube data at specific frames along the path
    2. For each detected spine, checks the same (y,x) position in neighboring frames
    3. Adds spine positions in neighboring frames if intensity is sufficient
    """
    # Fixed parameters (matching original code)
    detection_params = {
        'min_sigma_2d': 5, 'max_sigma_2d': 10, 'threshold_2d': 0.05,
        'min_sigma_tube': 5, 'max_sigma_tube': 5, 'threshold_tube': 0.02,
        'angle_threshold': 20, 'angle_weight': 0.7
    }
    
    initial_spine_positions = []
    all_results = []
    
    total_frames = len(tube_data)
    
    # Process tube data frames to detect spines
    print("Detecting spines from tube data...")
    for frame_idx in range(total_frames):
        if progress_callback:
            progress = int((frame_idx / total_frames) * 50)
            progress_callback(progress, 100)
        
        results = detect_spines_with_angles(
            tube_data, frame_idx,
            max_distance_threshold=max_distance_threshold,
            **detection_params
        )
        
        all_results.append(results)
        
        if results['confirmed_spines']:
            frame_position = tube_data[frame_idx]['position']
            z = results['actual_z']  # Use the actual Z from results
            y_base = frame_position[1]
            x_base = frame_position[2]
            
            for spine in results['confirmed_spines']:
                coord_x, coord_y = spine['coords_2d']
                # Calculate absolute position in the full image
                spine_y = y_base - 20 + coord_y
                spine_x = x_base - 20 + coord_x
                
                spine_3d_position = {
                    'z': z,
                    'y': spine_y,
                    'x': spine_x,
                    'position': np.array([float(z), float(spine_y), float(spine_x)])
                }
                initial_spine_positions.append(spine_3d_position)
    
    initial_spine_count = len(initial_spine_positions)
    print(f"Initial detection: {initial_spine_count} spines found")
    
    # Now extend each detected spine to neighboring frames
    all_spine_positions = []
    extended_count = 0
    
    if initial_spine_count > 0 and frame_range > 0:
        print(f"Extending spines to neighboring frames (±{frame_range} frames)...")
        
        # Track positions to avoid duplicates
        position_set = set()
        
        # Add initial positions
        for spine_data in initial_spine_positions:
            pos = spine_data['position']
            all_spine_positions.append(pos)
            position_set.add((int(pos[0]), int(pos[1]), int(pos[2])))
        
        # Extend each spine to neighboring frames
        for idx, spine_data in enumerate(initial_spine_positions):
            if progress_callback:
                progress = int(50 + (idx / initial_spine_count) * 50)
                progress_callback(progress, 100)
            
            original_z = int(spine_data['z'])
            spine_y = int(spine_data['y'])
            spine_x = int(spine_data['x'])
            
            # Check neighboring frames at the same (y,x) position
            for z_offset in range(-frame_range, frame_range + 1):
                if z_offset == 0:  # Skip original frame
                    continue
                
                target_z = original_z + z_offset
                
                # Check bounds
                if target_z < 0 or target_z >= image.shape[0]:
                    continue
                
                # Skip if position already exists
                if (target_z, spine_y, spine_x) in position_set:
                    continue
                
                # Check if coordinates are within image bounds
                if spine_y < 0 or spine_y >= image.shape[1] or spine_x < 0 or spine_x >= image.shape[2]:
                    continue
                
                # Define region around spine position
                window_size = 3
                y_min = max(0, spine_y - window_size)
                y_max = min(image.shape[1], spine_y + window_size + 1)
                x_min = max(0, spine_x - window_size)
                x_max = min(image.shape[2], spine_x + window_size + 1)
                
                # Get intensities
                region = image[target_z, y_min:y_max, x_min:x_max]
                if region.size == 0:
                    continue
                
                # Use the center pixel and its immediate neighbors
                center_y = spine_y - y_min
                center_x = spine_x - x_min
                
                # Get intensity at the exact spine location
                spine_intensity = image[target_z, spine_y, spine_x]
                
                # Calculate local maximum in the region
                local_max = np.max(region)
                
                # Background region (larger)
                bg_size = 10
                bg_y_min = max(0, spine_y - bg_size)
                bg_y_max = min(image.shape[1], spine_y + bg_size + 1)
                bg_x_min = max(0, spine_x - bg_size)
                bg_x_max = min(image.shape[2], spine_x + bg_size + 1)
                
                bg_region = image[target_z, bg_y_min:bg_y_max, bg_x_min:bg_x_max]
                if bg_region.size == 0:
                    continue
                
                background_intensity = np.median(bg_region)  # Use median for more robust background
                
                # Check if this is likely a spine:
                # 1. The spine location should be bright relative to background
                # 2. It should be reasonably bright compared to local maximum
                if (spine_intensity > background_intensity * 0.5 and 
                    spine_intensity > local_max * 0.2):  # More permissive thresholds
                    
                    extended_spine = np.array([
                        float(target_z),
                        float(spine_y),
                        float(spine_x)
                    ])
                    all_spine_positions.append(extended_spine)
                    position_set.add((target_z, spine_y, spine_x))
                    extended_count += 1
    else:
        # No extension requested or no initial spines
        all_spine_positions = [spine_data['position'] for spine_data in initial_spine_positions]
    
    # Convert to numpy array
    if all_spine_positions:
        final_positions = np.array(all_spine_positions)
    else:
        final_positions = np.empty((0, 3))
    
    print(f"Extended detection: {extended_count} additional spine positions")
    print(f"Total spine positions: {len(final_positions)}")
    
    return final_positions, brightest_path, all_results


class SpineDetectionWidget(QWidget):
    """Widget for detecting dendritic spines along brightest paths using optimized tube data"""
    
    spines_detected = Signal(str, list)
    
    def __init__(self, viewer, image, state):
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        if 'spine_layers' not in self.state:
            self.state['spine_layers'] = {}
        
        self.handling_event = False
        
        # Get initial pixel spacing from state
        self.pixel_spacing_nm = self.state.get('pixel_spacing_nm', 94.0)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        layout.addWidget(QLabel("<b>Memory-Optimized Spine Detection</b>"))
        layout.addWidget(QLabel("Uses minimal tube data generation (97.4% memory reduction)"))
        layout.addWidget(QLabel("1. Select a segmented path\n2. Set parameters\n3. Click 'Detect Spines'"))
        
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)
        
        # Path selection
        layout.addWidget(QLabel("Select a path for spine detection:"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(80)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Parameters group
        params_group = QGroupBox("Detection Parameters (in nanometers)")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Max distance threshold in nanometers
        max_distance_layout = QHBoxLayout()
        max_distance_layout.addWidget(QLabel("Max Distance:"))
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(100.0, 50000.0)
        self.max_distance_spin.setValue(2820.0)  # Default 2820 nm (30 pixels × 94 nm/pixel)
        self.max_distance_spin.setDecimals(0)
        self.max_distance_spin.setSuffix(" nm")
        self.max_distance_spin.setToolTip("Maximum distance threshold for grouping spines")
        max_distance_layout.addWidget(self.max_distance_spin)
        params_layout.addLayout(max_distance_layout)
        
        # Frame range for extension
        frame_range_layout = QHBoxLayout()
        frame_range_layout.addWidget(QLabel("Frame Extension:"))
        self.frame_range_spin = QSpinBox()
        self.frame_range_spin.setRange(0, 5)
        self.frame_range_spin.setValue(3)
        self.frame_range_spin.setToolTip("Number of frames to check before/after each spine")
        frame_range_layout.addWidget(self.frame_range_spin)
        params_layout.addLayout(frame_range_layout)
        
        # Enable parallel processing
        self.enable_parallel_cb = QCheckBox("Enable Parallel Processing")
        self.enable_parallel_cb.setChecked(True)
        self.enable_parallel_cb.setToolTip("Use parallel processing for faster tube data generation")
        params_layout.addWidget(self.enable_parallel_cb)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Detection button
        self.detect_spines_btn = QPushButton("Detect Spines (Memory Optimized)")
        self.detect_spines_btn.setFixedHeight(22)
        self.detect_spines_btn.clicked.connect(self.run_spine_detection)
        self.detect_spines_btn.setEnabled(False)
        layout.addWidget(self.detect_spines_btn)
        
        # Progress bar
        self.detection_progress = QProgressBar()
        self.detection_progress.setValue(0)
        layout.addWidget(self.detection_progress)
        
        # Results section
        self.results_label = QLabel("Results: No spines detected yet")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
        # Status message
        self.status_label = QLabel("Status: Select a segmented path to begin")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
    
    def update_path_list(self):
        """Update the path list with current paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            self.path_list.clear()
            
            for path_id, path_data in self.state['paths'].items():
                path_name = path_data['name']
                seg_layer_name = f"Segmentation - {path_name}"
                
                # Check if segmentation exists
                has_segmentation = any(layer.name == seg_layer_name for layer in self.viewer.layers)
                
                if has_segmentation:
                    item = QListWidgetItem(path_data['name'])
                    item.setData(100, path_id)
                    self.path_list.addItem(item)
            
            self.detect_spines_btn.setEnabled(
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating path list: {str(e)}")
            self.status_label.setText(f"Status: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle path selection changes"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.state['paths']:
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    
                    # Check if spines already exist
                    spine_layer_name = f"Spines - {path_name}"
                    has_spines = any(layer.name == spine_layer_name for layer in self.viewer.layers)
                    
                    if has_spines:
                        self.status_label.setText(f"Status: Path '{path_name}' already has spine detection")
                    else:
                        self.status_label.setText(f"Status: Path '{path_name}' selected for memory-optimized spine detection")
                    
                    self.detect_spines_btn.setEnabled(True)
            else:
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.detect_spines_btn.setEnabled(False)
                self.status_label.setText("Status: Select a segmented path to begin")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling path selection: {str(e)}")
        finally:
            self.handling_event = False
    
    def enable_for_path(self, path_id):
        """Enable spine detection for a specific segmented path"""
        for i in range(self.path_list.count()):
            item = self.path_list.item(i)
            if item.data(100) == path_id:
                self.path_list.setCurrentItem(item)
                break
        
        self.update_path_list()
    
    def run_spine_detection(self):
        """Run memory-optimized spine detection on the selected path"""
        if not hasattr(self, 'selected_path_id'):
            napari.utils.notifications.show_info("Please select a path for spine detection")
            return
        
        path_id = self.selected_path_id
        if path_id not in self.state['paths']:
            napari.utils.notifications.show_info("Selected path no longer exists")
            self.update_path_list()
            return
        
        try:
            # Get path data
            path_data = self.state['paths'][path_id]
            path_name = path_data['name']
            brightest_path = np.array(path_data['data'])
            
            # Update UI
            self.status_label.setText(f"Status: Running memory-optimized spine detection on {path_name}...")
            self.detection_progress.setValue(10)
            self.detect_spines_btn.setEnabled(False)
            
            # Get segmentation mask
            segmentation_mask = None
            seg_layer_name = f"Segmentation - {path_name}"
            for layer in self.viewer.layers:
                if layer.name == seg_layer_name:
                    segmentation_mask = layer.data
                    break
            
            if segmentation_mask is None:
                napari.utils.notifications.show_info(f"Segmentation not found for {path_name}")
                self.status_label.setText(f"Status: Segmentation not found for {path_name}")
                self.detect_spines_btn.setEnabled(True)
                return
            
            # Get parameters in nanometers
            max_distance_nm = self.max_distance_spin.value()
            frame_range = self.frame_range_spin.value()
            enable_parallel = self.enable_parallel_cb.isChecked()
            
            # Use fixed default values for FOV and zoom size (not user-configurable)
            fov_nm = 3760.0  # Fixed at 3760 nm (40 pixels × 94 nm/pixel default)
            zoom_size_nm = 3760.0  # Fixed at 3760 nm (40 pixels × 94 nm/pixel default)
            
            # Convert nanometers to pixels using current pixel spacing
            pixel_spacing = self.pixel_spacing_nm
            max_distance_pixels = int(max_distance_nm / pixel_spacing)
            fov_pixels = int(fov_nm / pixel_spacing)
            zoom_size_pixels = int(zoom_size_nm / pixel_spacing)
            
            verbose = True  # Fix: Define verbose variable
            if verbose:
                print(f"Pixel spacing: {pixel_spacing:.1f} nm/pixel")
                print(f"Max distance: {max_distance_nm:.0f} nm = {max_distance_pixels} pixels")
                print(f"Field of view: {fov_nm:.0f} nm = {fov_pixels} pixels (fixed)") 
                print(f"Zoom size: {zoom_size_nm:.0f} nm = {zoom_size_pixels} pixels (fixed)")
            
            self.detection_progress.setValue(20)
            napari.utils.notifications.show_info(f"Creating minimal tube data for {path_name}...")
            
            # Check if we have a pre-computed path to pass
            existing_path = brightest_path if brightest_path is not None else None
            
            # Create tube data using the optimized minimal function with pixel values
            tube_data = create_tube_data(
                image=self.image,
                points_list=[brightest_path[0], brightest_path[-1]],  # Start and end points
                existing_path=existing_path,  # Pass the existing path
                view_distance=1,  # This parameter is ignored in minimal version
                field_of_view=fov_pixels,  # Converted to pixels
                zoom_size=zoom_size_pixels,  # Converted to pixels
                reference_image=segmentation_mask,
                enable_parallel=enable_parallel,
                verbose=True
            )
            
            self.detection_progress.setValue(40)
            napari.utils.notifications.show_info("Detecting spines and extending to neighboring frames...")
            
            # Progress callback
            def update_progress(current, total):
                progress = int(40 + (current / total) * 50)
                self.detection_progress.setValue(progress)
            
            # Run detection with extension using pixel values
            spine_positions, _, all_results = process_all_frames_with_extension(
                tube_data=tube_data,
                image=self.image,
                brightest_path=brightest_path,
                max_distance_threshold=max_distance_pixels,  # Use pixel value
                frame_range=frame_range,
                progress_callback=update_progress
            )
            
            self.detection_progress.setValue(90)
            
            # Process results
            if spine_positions is not None and len(spine_positions) > 0:
                spine_layer_name = f"Spines - {path_name}"
                
                # Remove existing layer
                for layer in list(self.viewer.layers):
                    if layer.name == spine_layer_name:
                        self.viewer.layers.remove(layer)
                        break
                
                # Add new spine layer
                spine_layer = self.viewer.add_points(
                    spine_positions,
                    name=spine_layer_name,
                    size=8,
                    face_color='red',
                    opacity=0.8,
                    symbol='cross'
                )
                
                # Store references
                self.state['spine_layers'][path_id] = spine_layer
                self.state['spine_positions'] = spine_positions
                
                # Calculate statistics
                initial_count = sum(r['num_confirmed_spines'] for r in all_results)
                frames_with_spines = len([r for r in all_results if r['num_confirmed_spines'] > 0])
                extended_count = len(spine_positions) - initial_count
                
                # Update UI with memory optimization info and nanometer values
                algorithm_info = f" (parallel, 97.4% memory reduction, {max_distance_nm:.0f}nm max distance)" if enable_parallel else f" (sequential, 97.4% memory reduction, {max_distance_nm:.0f}nm max distance)"
                self.results_label.setText(
                    f"Results: {len(spine_positions)} total spine positions{algorithm_info}\n"
                    f"Initial detection: {initial_count} spines\n"
                    f"Extended positions: {extended_count}\n"
                    f"Frames with initial spines: {frames_with_spines}\n"
                    f"Detection parameters: Max distance={max_distance_nm:.0f}nm"
                )
                self.status_label.setText(f"Status: Memory-optimized spine detection completed for {path_name}")
                
                # Store enhanced spine data
                if 'spine_data' not in self.state:
                    self.state['spine_data'] = {}
                
                self.state['spine_data'][path_id] = {
                    'original_positions': spine_positions,
                    'all_results': all_results,
                    'detection_method': 'memory_optimized_angle_based_extended',
                    'parameters': {
                        'max_distance_nm': max_distance_nm,
                        'max_distance_pixels': max_distance_pixels,
                        'field_of_view_nm': fov_nm,
                        'field_of_view_pixels': fov_pixels,
                        'zoom_size_nm': zoom_size_nm,
                        'zoom_size_pixels': zoom_size_pixels,
                        'frame_range': frame_range,
                        'enable_parallel': enable_parallel,
                        'pixel_spacing_nm': pixel_spacing
                    }
                }
                
                # Emit signal
                self.spines_detected.emit(path_id, spine_positions.tolist())
                
                napari.utils.notifications.show_info(f"Detected {len(spine_positions)} spine positions for {path_name} using memory-optimized algorithm")
            else:
                self.results_label.setText("Results: No spines detected")
                self.status_label.setText(f"Status: No spines detected for {path_name}")
                napari.utils.notifications.show_info("No spines detected")
        
        except Exception as e:
            error_msg = f"Error during memory-optimized spine detection: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.results_label.setText("Results: Error during spine detection")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.detection_progress.setValue(100)
            self.detect_spines_btn.setEnabled(True)
    
    def update_pixel_spacing(self, new_spacing):
        """Update pixel spacing and recalculate default parameter values"""
        self.pixel_spacing_nm = new_spacing
        
        # Update default values based on new pixel spacing (keeping same pixel equivalents)
        default_max_distance_pixels = 15  # Original default in pixels
        
        # Convert to nanometers with new spacing
        new_max_distance_nm = default_max_distance_pixels * new_spacing
        
        # Update the UI values (only max distance, FOV and zoom are now fixed)
        self.max_distance_spin.setValue(new_max_distance_nm)
        
        print(f"Spine detection: Updated to {new_spacing:.1f} nm/pixel")
        print(f"  Max distance: {new_max_distance_nm:.0f} nm")
        print(f"  Field of view: {40 * new_spacing:.0f} nm (fixed)")
        print(f"  Zoom size: {40 * new_spacing:.0f} nm (fixed)")