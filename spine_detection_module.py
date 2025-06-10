import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox
)
from qtpy.QtCore import Signal
import sys
sys.path.append('../path_tracing/brightest-path-lib')
from brightest_path_lib.visualization import create_tube_data

# Import image processing libraries
from scipy import ndimage as ndi
from skimage import filters, segmentation, morphology, measure, feature


def filter_dendrites_by_spine_symmetry(all_detections, dendrite_path, frame_data, 
                                     symmetry_window=5, min_spine_pairs=2):
    """
    Filter parallel dendrites by checking if detections appear symmetrically 
    on both sides of the dendrite shaft.
    """
    
    def get_side_of_dendrite(detection, center_x, center_y, forward_vector):
        """Determine which side of the dendrite (left/right) a detection is on."""
        dx = detection['plane_x'] - center_x
        dy = detection['plane_y'] - center_y
        detection_vector = np.array([dx, dy])
        
        if len(forward_vector) == 3:
            perp_vector = np.array([-forward_vector[2], forward_vector[1]])
        else:
            perp_vector = np.array([-forward_vector[1], forward_vector[0]])
        
        side_value = np.dot(detection_vector, perp_vector)
        return 'right' if side_value > 0 else 'left'
    
    def find_symmetric_pairs(frame_detections, center_x, center_y, forward_vector):
        """Find pairs of detections that are symmetric across the dendrite shaft."""
        if len(frame_detections) < 2:
            return []
        
        left_detections = []
        right_detections = []
        
        for det in frame_detections:
            side = get_side_of_dendrite(det, center_x, center_y, forward_vector)
            if side == 'left':
                left_detections.append(det)
            else:
                right_detections.append(det)
        
        symmetric_pairs = []
        for left_det in left_detections:
            for right_det in right_detections:
                left_dist = np.sqrt((left_det['plane_x'] - center_x)**2 + 
                                  (left_det['plane_y'] - center_y)**2)
                right_dist = np.sqrt((right_det['plane_x'] - center_x)**2 + 
                                   (right_det['plane_y'] - center_y)**2)
                
                dist_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
                
                left_angle = np.arctan2(left_det['plane_y'] - center_y, 
                                      left_det['plane_x'] - center_x)
                right_angle = np.arctan2(right_det['plane_y'] - center_y, 
                                       right_det['plane_x'] - center_x)
                
                angle_diff = abs(abs(left_angle - right_angle) - np.pi)
                
                if dist_ratio > 0.7 and angle_diff < np.pi/4:
                    symmetric_pairs.append((left_det, right_det))
        
        return symmetric_pairs
    
    # Analyze symmetry across all frames
    frame_symmetry_scores = []
    all_symmetric_detections = set()
    
    for frame_idx, frame in enumerate(frame_data):
        if 'spines' not in frame or len(frame['spines']) == 0:
            frame_symmetry_scores.append(0)
            continue
        
        center_x = frame['normal_plane'].shape[1] // 2
        center_y = frame['normal_plane'].shape[0] // 2
        forward_vector = frame['basis_vectors']['forward']
        
        symmetric_pairs = find_symmetric_pairs(frame['spines'], center_x, center_y, forward_vector)
        
        for left_det, right_det in symmetric_pairs:
            left_id = (left_det['frame_idx'], left_det['plane_x'], left_det['plane_y'])
            right_id = (right_det['frame_idx'], right_det['plane_x'], right_det['plane_y'])
            all_symmetric_detections.add(left_id)
            all_symmetric_detections.add(right_id)
        
        total_detections = len(frame['spines'])
        symmetric_detections = len(symmetric_pairs) * 2
        symmetry_score = symmetric_detections / total_detections if total_detections > 0 else 0
        frame_symmetry_scores.append(symmetry_score)
    
    # Analyze temporal patterns for each detection
    filtered_spines = []
    filtered_dendrites = []
    
    for detection in all_detections:
        frame_idx = detection['frame_idx']
        det_id = (frame_idx, detection['plane_x'], detection['plane_y'])
        
        is_symmetric = det_id in all_symmetric_detections
        
        # Check symmetry in nearby frames
        start_frame = max(0, frame_idx - symmetry_window//2)
        end_frame = min(len(frame_symmetry_scores), frame_idx + symmetry_window//2 + 1)
        
        nearby_symmetry_scores = frame_symmetry_scores[start_frame:end_frame]
        avg_nearby_symmetry = np.mean(nearby_symmetry_scores) if nearby_symmetry_scores else 0
        
        # Classification logic
        is_likely_dendrite = False
        
        # Strong dendrite indicators
        if detection['area'] > 100 and not is_symmetric and avg_nearby_symmetry < 0.3:
            is_likely_dendrite = True
        
        if (detection.get('aspect_ratio', 1) > 3.0 and 
            detection['area'] > 60 and 
            avg_nearby_symmetry < 0.4):
            is_likely_dendrite = True
        
        if frame_idx < len(frame_data):
            frame = frame_data[frame_idx]
            center_x = frame['normal_plane'].shape[1] // 2
            center_y = frame['normal_plane'].shape[0] // 2
            forward_vector = frame['basis_vectors']['forward']
            
            detection_side = get_side_of_dendrite(detection, center_x, center_y, forward_vector)
            
            opposite_side_count = 0
            same_side_count = 0
            
            for i in range(max(0, frame_idx-3), min(len(frame_data), frame_idx+4)):
                if i < len(frame_data) and 'spines' in frame_data[i]:
                    frame_center_x = frame_data[i]['normal_plane'].shape[1] // 2
                    frame_center_y = frame_data[i]['normal_plane'].shape[0] // 2
                    frame_forward = frame_data[i]['basis_vectors']['forward']
                    
                    for other_det in frame_data[i]['spines']:
                        other_side = get_side_of_dendrite(other_det, frame_center_x, 
                                                        frame_center_y, frame_forward)
                        if other_side == detection_side:
                            same_side_count += 1
                        else:
                            opposite_side_count += 1
            
            if (same_side_count >= 3 and 
                opposite_side_count <= 1 and 
                detection['area'] > 40):
                is_likely_dendrite = True
        
        if (detection['distance'] > 15 and 
            detection['distance'] < 40 and 
            not is_symmetric and
            detection['area'] > 50):
            is_likely_dendrite = True
        
        # Classify the detection
        if is_likely_dendrite:
            filtered_dendrites.append(detection)
        else:
            filtered_spines.append(detection)
    
    return filtered_spines, filtered_dendrites


def detect_spines_with_geometric_filtering(tube_data, shaft_radius=6, max_spine_distance=30, 
                                         min_spine_score=0.1, show_visualization=False):
    """Enhanced spine detection with geometric dendrite filtering.
    
    Parameters:
    -----------
    tube_data : list
        Tube data from create_tube_data()
    shaft_radius : int, default=6
        Dendrite shaft radius in pixels
    max_spine_distance : int, default=30  
        Maximum distance from center to search for spines
    min_spine_score : float, default=0.1
        Minimum score threshold (lower = more permissive)
    show_visualization : bool, default=False
        Whether to show matplotlib visualization
    """
    
    # Set reasonable defaults for internal parameters
    min_spine_intensity = 1.2
    min_spine_area = 5
    distance_threshold = 10
    
    dendrite_path = np.array([frame['current_point_in_path'] for frame in tube_data])
    
    # Visualization setup (only if requested)
    if show_visualization:
        import matplotlib.pyplot as plt
        num_frames = min(10, len(tube_data))
        fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
        if num_frames == 1:
            axes = [axes]
        frame_indices = np.linspace(0, len(tube_data)-1, num_frames, dtype=int)
    
    spine_candidates = []
    frame_data = []
    
    # Process each frame
    for frame_idx, frame in enumerate(tube_data):
        normal_plane = frame['normal_plane']
        current_position = frame['current_point_in_path']
        
        if np.max(normal_plane) < 0.1:
            frame_result = {
                'frame_idx': frame_idx,
                'position': current_position,
                'normal_plane': normal_plane,
                'dendrite_binary': np.zeros_like(normal_plane, dtype=bool),
                'spine_regions': np.zeros_like(normal_plane, dtype=bool),
                'segments': np.zeros_like(normal_plane, dtype=int),
                'spines': [],
                'basis_vectors': frame['basis_vectors']
            }
            frame_data.append(frame_result)
            continue
        
        up = frame['basis_vectors']['up']
        right = frame['basis_vectors']['right']
        forward = frame['basis_vectors']['forward']
        
        center_y, center_x = normal_plane.shape[0] // 2, normal_plane.shape[1] // 2
        
        # Preprocessing
        smoothed = filters.gaussian(normal_plane, sigma=1.5)
        y_coords, x_coords = np.ogrid[:normal_plane.shape[0], :normal_plane.shape[1]]
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        if max_spine_distance is not None:
            mask_too_far = dist_from_center > max_spine_distance
            smoothed[mask_too_far] = 0
        
        # Watershed segmentation
        markers = np.zeros_like(smoothed, dtype=int)
        markers[dist_from_center < shaft_radius] = 1
        
        try:
            thresh_value = filters.threshold_otsu(smoothed)
        except:
            thresh_value = 0.1
        
        dendrite_binary = filters.apply_hysteresis_threshold(
            smoothed, low=thresh_value * 0.5, high=thresh_value * 0.75
        )
        
        if np.sum(dendrite_binary) < 10:
            continue
        
        dendrite_binary = morphology.remove_small_objects(dendrite_binary, min_size=min_spine_area)
        dendrite_binary = morphology.binary_closing(dendrite_binary, morphology.disk(2))
        
        # Find main dendrite component
        labeled_dendrite = measure.label(dendrite_binary)
        if labeled_dendrite[center_y, center_x] > 0:
            main_label = labeled_dendrite[center_y, center_x]
            dendrite_binary = labeled_dendrite == main_label
        else:
            props = measure.regionprops(labeled_dendrite)
            if props:
                min_dist = float('inf')
                closest_label = 0
                for prop in props:
                    y, x = prop.centroid
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = prop.label
                dendrite_binary = labeled_dendrite == closest_label
        
        # Mark spine candidates
        outer_mask = dendrite_binary & (dist_from_center > shaft_radius * 1.2)
        spine_candidates_binary = (smoothed > thresh_value * min_spine_intensity) & outer_mask
        spine_candidates_binary = morphology.remove_small_objects(spine_candidates_binary, min_size=min_spine_area)
        markers[spine_candidates_binary] = 2
        
        # Apply watershed
        elevation_map = -smoothed
        segments = segmentation.watershed(elevation_map, markers, mask=dendrite_binary)
        spine_regions = segments == 2
        
        # Label and measure
        labeled_spines = measure.label(spine_regions)
        spine_props = measure.regionprops(labeled_spines, intensity_image=smoothed)
        
        med_intensity = np.median(smoothed[dendrite_binary])
        max_intensity = np.max(smoothed)
        
        frame_spines = []
        
        for prop_idx, prop in enumerate(spine_props):
            if prop.area < min_spine_area:
                continue
            
            y_centroid, x_centroid = prop.weighted_centroid if hasattr(prop, 'weighted_centroid') else prop.centroid
            
            # Calculate 3D position
            dx_spine = x_centroid - center_x
            dy_spine = y_centroid - center_y
            spine_position = current_position + dx_spine * right + dy_spine * up
            
            dist = np.sqrt((x_centroid - center_x)**2 + (y_centroid - center_y)**2)
            
            if max_spine_distance is not None and dist > max_spine_distance:
                continue
            
            # Basic shape analysis
            if hasattr(prop, 'axis_major_length') and hasattr(prop, 'axis_minor_length'):
                major_axis = prop.axis_major_length
                minor_axis = prop.axis_minor_length
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 10
            else:
                minr, minc, maxr, maxc = prop.bbox
                height = maxr - minr
                width = maxc - minc
                aspect_ratio = max(height, width) / max(1, min(height, width))
            
            # Scoring
            intensity_factor = (prop.mean_intensity - med_intensity) / (max_intensity - med_intensity) if max_intensity > med_intensity else 0
            size_factor = min(1.0, prop.area / 30)
            distance_factor = min(1.0, dist / (shaft_radius * 2))
            spine_score = intensity_factor * 0.4 + size_factor * 0.3 + distance_factor * 0.3
            
            if spine_score < min_spine_score:
                continue
            
            spine_info = {
                'position': spine_position,
                'frame_idx': frame_idx,
                'plane_x': x_centroid,
                'plane_y': y_centroid,
                'distance': dist,
                'intensity': prop.mean_intensity,
                'area': prop.area,
                'score': spine_score,
                'aspect_ratio': aspect_ratio
            }
            
            spine_candidates.append(spine_info)
            frame_spines.append(spine_info)
        
        frame_result = {
            'frame_idx': frame_idx,
            'position': current_position,
            'normal_plane': normal_plane,
            'dendrite_binary': dendrite_binary,
            'spine_regions': spine_regions,
            'segments': segments,
            'spines': frame_spines,
            'basis_vectors': {'up': up, 'right': right, 'forward': forward}
        }
        frame_data.append(frame_result)
    
    # Apply geometric filtering
    print("Applying geometric symmetry-based filtering...")
    filtered_spines, filtered_dendrites = filter_dendrites_by_spine_symmetry(
        spine_candidates, dendrite_path, frame_data
    )
    
    print(f"Original detections: {len(spine_candidates)}")
    print(f"Filtered as dendrites: {len(filtered_dendrites)}")
    print(f"Remaining spines: {len(filtered_spines)}")
    
    # Visualization (only if requested)
    if show_visualization:
        for frame_idx in frame_indices:
            if frame_idx < len(frame_data):
                ax_idx = np.where(frame_indices == frame_idx)[0][0]
                ax = axes[ax_idx]
                
                frame = frame_data[frame_idx]
                ax.imshow(frame['normal_plane'], cmap='gray')
                
                center_x = frame['normal_plane'].shape[1] // 2
                center_y = frame['normal_plane'].shape[0] // 2
                
                ax.plot(center_x, center_y, 'b+', markersize=10)
                circle = plt.Circle((center_x, center_y), shaft_radius, 
                                  color='blue', fill=False, linewidth=1)
                ax.add_patch(circle)
                
                for spine in filtered_spines:
                    if spine['frame_idx'] == frame_idx:
                        ax.plot(spine['plane_x'], spine['plane_y'], 'go', markersize=8, 
                               markeredgecolor='white', label='Spine')
                
                for dendrite in filtered_dendrites:
                    if dendrite['frame_idx'] == frame_idx:
                        ax.plot(dendrite['plane_x'], dendrite['plane_y'], 'ro', markersize=8, 
                               markeredgecolor='white', label='Dendrite')
                
                ax.set_title(f'Frame {frame_idx}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Group nearby spine candidates
    spine_positions = []
    processed = set()
    sorted_candidates = sorted(filtered_spines, key=lambda x: x['score'], reverse=True)
    
    for i, candidate in enumerate(sorted_candidates):
        if i in processed:
            continue
        
        pos_i = candidate['position']
        group = [candidate]
        processed.add(i)
        
        for j, other in enumerate(sorted_candidates):
            if j in processed or i == j:
                continue
            
            pos_j = other['position']
            if np.linalg.norm(pos_i - pos_j) < distance_threshold:
                group.append(other)
                processed.add(j)
        
        best = max(group, key=lambda x: x['score'])
        spine_positions.append(best['position'])
    
    print(f"Final spine count after grouping: {len(spine_positions)}")
    
    return spine_positions, dendrite_path, frame_data


class SpineDetectionWidget(QWidget):
    """Widget for detecting dendritic spines along brightest paths"""
    
    spines_detected = Signal(str, list)
    
    def __init__(self, viewer, image, state):
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        if 'spine_layers' not in self.state:
            self.state['spine_layers'] = {}
        
        self.handling_event = False
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        layout.addWidget(QLabel("<b>Enhanced Dendritic Spine Detection</b>"))
        layout.addWidget(QLabel("1. Select a segmented path\n2. Configure detection parameters\n3. Click 'Detect Spines' to find spines"))
        
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
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Shaft radius
        shaft_radius_layout = QHBoxLayout()
        shaft_radius_layout.addWidget(QLabel("Shaft Radius:"))
        self.shaft_radius_spin = QSpinBox()
        self.shaft_radius_spin.setRange(2, 30)
        self.shaft_radius_spin.setValue(6)
        self.shaft_radius_spin.setToolTip("Approximate radius of the dendrite shaft")
        shaft_radius_layout.addWidget(self.shaft_radius_spin)
        params_layout.addLayout(shaft_radius_layout)
        
        # Maximum spine distance (search area)
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Max Search Area:"))
        self.max_distance_spin = QSpinBox()
        self.max_distance_spin.setRange(10, 50)
        self.max_distance_spin.setValue(30)
        self.max_distance_spin.setToolTip("Maximum distance from dendrite center to search for spines")
        distance_layout.addWidget(self.max_distance_spin)
        params_layout.addLayout(distance_layout)
        
        # Minimum spine score
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("Min Spine Score:"))
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.01, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.1)
        self.score_spin.setToolTip("Minimum score threshold for spine detection (lower = more permissive)")
        score_layout.addWidget(self.score_spin)
        params_layout.addLayout(score_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Detection button
        self.detect_spines_btn = QPushButton("Detect Spines")
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
                
                has_segmentation = False
                for layer in self.viewer.layers:
                    if layer.name == seg_layer_name:
                        has_segmentation = True
                        break
                
                if has_segmentation:
                    item = QListWidgetItem(path_data['name'])
                    item.setData(100, path_id)
                    self.path_list.addItem(item)
            
            self.detect_spines_btn.setEnabled(
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating spine detection path list: {str(e)}")
            self.status_label.setText(f"Status: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle when path selection changes in the list"""
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
                    
                    spine_layer_name = f"Spines - {path_name}"
                    has_spines = False
                    for layer in self.viewer.layers:
                        if layer.name == spine_layer_name:
                            has_spines = True
                            break
                    
                    if has_spines:
                        self.status_label.setText(f"Status: Path '{path_name}' already has spine detection")
                    else:
                        self.status_label.setText(f"Status: Path '{path_name}' selected for enhanced spine detection")
                    
                    self.detect_spines_btn.setEnabled(True)
            else:
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.detect_spines_btn.setEnabled(False)
                self.status_label.setText("Status: Select a segmented path to begin")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling spine detection path selection: {str(e)}")
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
    
    def project_spine_positions(self, spine_positions, projection_range):
        """Project spine positions across frames to make them visible in a range of frames."""
        if isinstance(spine_positions, list) and len(spine_positions) == 0:
            return []
        if isinstance(spine_positions, np.ndarray) and spine_positions.size == 0:
            return []
            
        if not isinstance(spine_positions, np.ndarray):
            spine_positions = np.array(spine_positions)
        
        # Group spine positions by their spatial (y, x) coordinates
        spine_groups = {}
        for spine in spine_positions:
            key = (spine[1], spine[2])
            if key not in spine_groups:
                spine_groups[key] = []
            spine_groups[key].append(spine[0])
        
        # Create projected spine positions
        projected_spines = []
        
        for (y, x), z_coords in spine_groups.items():
            min_z = min(z_coords)
            max_z = max(z_coords)
            
            extended_min_z = max(0, min_z - projection_range)
            extended_max_z = min(len(self.image) - 1, max_z + projection_range)
            
            for z in range(int(extended_min_z), int(extended_max_z) + 1):
                projected_spines.append([z, float(y), float(x)])
        
        if len(projected_spines) > 0:
            return np.array(projected_spines)
        else:
            return np.empty((0, 3))
    
    def run_spine_detection(self):
        """Run enhanced spine detection on the selected path"""
        if not hasattr(self, 'selected_path_id'):
            napari.utils.notifications.show_info("Please select a path for spine detection")
            return
        
        path_id = self.selected_path_id
        if path_id not in self.state['paths']:
            napari.utils.notifications.show_info("Selected path no longer exists")
            self.update_path_list()
            return
        
        try:
            # Get the path data
            path_data = self.state['paths'][path_id]
            path_name = path_data['name']
            brightest_path = path_data['data'].copy()
            start_point = path_data['start'].copy() if 'start' in path_data else None
            end_point = path_data['end'].copy() if 'end' in path_data else None
            waypoints = [wp.copy() for wp in path_data['waypoints']] if 'waypoints' in path_data and path_data['waypoints'] else []
            
            # Update UI
            self.status_label.setText(f"Status: Running enhanced spine detection on {path_name}...")
            self.detection_progress.setValue(10)
            self.detect_spines_btn.setEnabled(False)
            
            # Get segmentation mask for this path
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
            
            # Get parameters from UI (only the 3 essential ones)
            params = {
                'shaft_radius': self.shaft_radius_spin.value(),
                'max_spine_distance': self.max_distance_spin.value(),
                'min_spine_score': self.score_spin.value()
            }
            
            self.detection_progress.setValue(20)
            napari.utils.notifications.show_info(f"Creating tubular data for {path_name}...")
            
            # Create tubular data for spine detection
            view_distance = 1
            field_of_view = 50
            zoom_size = 50
            
            tube_data = create_tube_data(
                image=self.image,
                start_point=start_point,
                goal_point=end_point,
                waypoints=waypoints if waypoints else None,
                reference_image=segmentation_mask,
                view_distance=view_distance,
                field_of_view=field_of_view,
                zoom_size=zoom_size
            )
            
            self.detection_progress.setValue(80)
            napari.utils.notifications.show_info("Running enhanced spine detection with geometric filtering...")
            
            # Run the enhanced detection algorithm
            spine_positions, dendrite_path, frame_data = detect_spines_with_geometric_filtering(
                tube_data=tube_data,
                shaft_radius=params['shaft_radius'],
                max_spine_distance=params['max_spine_distance'],
                min_spine_score=params['min_spine_score']
            )
            
            self.detection_progress.setValue(90)
            
            # Process the results
            if spine_positions is not None and len(spine_positions) > 0:
                spine_layer_name = f"Spines - {path_name}"
                
                # Remove existing layer if it exists
                for layer in list(self.viewer.layers):
                    if layer.name == spine_layer_name:
                        self.viewer.layers.remove(layer)
                        break
                
                spine_positions_array = np.array(spine_positions)
                
                # Always project spines across frames for better visibility  
                displayed_spine_positions = self.project_spine_positions(
                    spine_positions_array, 
                    projection_range=2  # Fixed projection range
                )
                
                # Add the new spine layer
                spine_layer = self.viewer.add_points(
                    displayed_spine_positions,
                    name=spine_layer_name,
                    size=8,
                    face_color='red',
                    opacity=0.8,
                    symbol='cross'
                )
                
                # Store spine layer reference
                self.state['spine_layers'][path_id] = spine_layer
                self.state['spine_positions'] = spine_positions
                
                # Update UI
                self.results_label.setText(f"Results: Detected {len(spine_positions)} spines for {path_name}")
                self.status_label.setText(f"Status: Spine detection completed for {path_name}")
                
                # Store enhanced spine data in state
                if 'spine_data' not in self.state:
                    self.state['spine_data'] = {}
                    
                self.state['spine_data'][path_id] = {
                    'original_positions': spine_positions_array,
                    'projected_positions': displayed_spine_positions,
                    'projection_range': 2,  # Fixed projection range
                    'frame_data': frame_data,
                    'detection_method': 'enhanced_geometric'
                }
                
                # Emit signal that spines were detected
                self.spines_detected.emit(path_id, spine_positions)
                
                napari.utils.notifications.show_info(f"Detected {len(spine_positions)} spines for {path_name}")
            else:
                self.results_label.setText("Results: No spines detected")
                self.status_label.setText(f"Status: No spines detected for {path_name}")
                napari.utils.notifications.show_info("No spines detected")
        
        except Exception as e:
            error_msg = f"Error during spine detection: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.results_label.setText("Results: Error during spine detection")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.detection_progress.setValue(100)
            self.detect_spines_btn.setEnabled(True)