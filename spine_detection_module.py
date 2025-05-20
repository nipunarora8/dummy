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
from brightest_path_lib.algorithm.find_spines import detect_spines_with_watershed
from brightest_path_lib.visualization import create_tube_data


class SpineDetectionWidget(QWidget):
    """Widget for detecting dendritic spines along brightest paths"""
    
    # Define signals
    spines_detected = Signal(str, list)  # path_id, spine_positions
    
    def __init__(self, viewer, image, state):
        """Initialize the spine detection widget.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance
        image : numpy.ndarray
            3D or higher-dimensional image data
        state : dict
            Shared state dictionary between modules
        """
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        # Initialize spine detection state
        if 'spine_layers' not in self.state:
            self.state['spine_layers'] = {}
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Title and instructions
        layout.addWidget(QLabel("<b>Dendritic Spine Detection</b>"))
        layout.addWidget(QLabel("1. Select a segmented path\n2. Configure detection parameters\n3. Click 'Detect Spines' to find spines"))
        
        # Add separator
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
        self.shaft_radius_spin.setValue(8)
        self.shaft_radius_spin.setToolTip("Approximate radius of the dendrite shaft")
        shaft_radius_layout.addWidget(self.shaft_radius_spin)
        params_layout.addLayout(shaft_radius_layout)
        
        # Minimum spine intensity
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Min Spine Intensity:"))
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.1, 5.0)
        self.intensity_spin.setSingleStep(0.1)
        self.intensity_spin.setValue(0.8)
        self.intensity_spin.setToolTip("Minimum intensity multiplier for spine detection")
        intensity_layout.addWidget(self.intensity_spin)
        params_layout.addLayout(intensity_layout)
        
        # Minimum spine area
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("Min Spine Area:"))
        self.area_spin = QSpinBox()
        self.area_spin.setRange(1, 50)
        self.area_spin.setValue(5)
        self.area_spin.setToolTip("Minimum area (in pixels) for a region to be considered a spine")
        area_layout.addWidget(self.area_spin)
        params_layout.addLayout(area_layout)
        
        # Minimum spine score
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("Min Spine Score:"))
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.01, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.1)
        self.score_spin.setToolTip("Minimum score threshold for spine detection")
        score_layout.addWidget(self.score_spin)
        params_layout.addLayout(score_layout)
        
        # Maximum spine distance
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Max Spine Distance:"))
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(5, 50)
        self.distance_spin.setValue(15)
        self.distance_spin.setToolTip("Maximum distance from dendrite center for spine detection")
        distance_layout.addWidget(self.distance_spin)
        params_layout.addLayout(distance_layout)
        
        # Distance threshold for grouping
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Distance Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(5, 50)
        self.threshold_spin.setValue(15)
        self.threshold_spin.setToolTip("Distance threshold for grouping nearby spine detections")
        threshold_layout.addWidget(self.threshold_spin)
        params_layout.addLayout(threshold_layout)
        
        # Filter dendrites option
        self.filter_dendrites_cb = QCheckBox("Filter Parallel Dendrites")
        self.filter_dendrites_cb.setChecked(True)
        self.filter_dendrites_cb.setToolTip("Filter out false positives from parallel dendrites")
        params_layout.addWidget(self.filter_dendrites_cb)
        
        # Visualization options
        self.show_visualization_cb = QCheckBox("Show Detection Visualization")
        self.show_visualization_cb.setChecked(True)
        self.show_visualization_cb.setToolTip("Show visualization of spine detection process")
        params_layout.addWidget(self.show_visualization_cb)
        
        # Set the layout for the parameters group
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Detection button
        self.detect_spines_btn = QPushButton("Detect Spines")
        self.detect_spines_btn.setFixedHeight(22)
        self.detect_spines_btn.clicked.connect(self.run_spine_detection)
        self.detect_spines_btn.setEnabled(False)  # Disabled until a segmented path is selected
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
            
            # Clear current list
            self.path_list.clear()
            
            # Add paths to list that have segmentation
            for path_id, path_data in self.state['paths'].items():
                path_name = path_data['name']
                seg_layer_name = f"Segmentation - {path_name}"
                
                # Check if this path has segmentation
                has_segmentation = False
                for layer in self.viewer.layers:
                    if layer.name == seg_layer_name:
                        has_segmentation = True
                        break
                
                if has_segmentation:
                    item = QListWidgetItem(path_data['name'])
                    item.setData(100, path_id)  # Store path ID as custom data
                    self.path_list.addItem(item)
            
            # Enable/disable detect button based on path availability
            self.detect_spines_btn.setEnabled(
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating spine detection path list: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle when path selection changes in the list"""
        # Prevent processing during updates
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.state['paths']:
                    # Store the selected path ID for spine detection
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    
                    # Check if this path has already had spine detection
                    spine_layer_name = f"Spines - {path_name}"
                    has_spines = False
                    for layer in self.viewer.layers:
                        if layer.name == spine_layer_name:
                            has_spines = True
                            break
                    
                    if has_spines:
                        self.status_label.setText(f"Status: Path '{path_name}' already has spine detection")
                    else:
                        self.status_label.setText(f"Status: Path '{path_name}' selected for spine detection")
                    
                    # Enable the detection button
                    self.detect_spines_btn.setEnabled(True)
            else:
                # No path selected
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
        # Select this path in the list if it exists
        for i in range(self.path_list.count()):
            item = self.path_list.item(i)
            if item.data(100) == path_id:
                self.path_list.setCurrentItem(item)
                break
        
        # If path isn't in the list, update the list (may be newly segmented)
        self.update_path_list()
    
    def run_spine_detection(self):
        """Run spine detection on the selected path"""
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
            brightest_path = path_data['data'].copy()  # Make a copy to ensure we don't modify the original
            start_point = path_data['start'].copy() if 'start' in path_data else None
            end_point = path_data['end'].copy() if 'end' in path_data else None
            waypoints = [wp.copy() for wp in path_data['waypoints']] if 'waypoints' in path_data and path_data['waypoints'] else []
            
            # Update UI
            self.status_label.setText(f"Status: Running spine detection on {path_name}...")
            self.detection_progress.setValue(10)
            self.detect_spines_btn.setEnabled(False)
            
            # Get segmentation mask for this path (if available)
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
            
            # Get parameters from UI
            params = {
                'shaft_radius': self.shaft_radius_spin.value(),
                'min_spine_intensity': self.intensity_spin.value(),
                'min_spine_area': self.area_spin.value(),
                'min_spine_score': self.score_spin.value(),
                'max_spine_distance': self.distance_spin.value(),
                'distance_threshold': self.threshold_spin.value(),
                'filter_dendrites': self.filter_dendrites_cb.isChecked(),
                'show_visualization': self.show_visualization_cb.isChecked()
            }
            
            self.detection_progress.setValue(20)
            napari.utils.notifications.show_info(f"Creating tubular data for {path_name}...")
            
            # Create tubular data for spine detection
            view_distance = 5
            field_of_view = 50
            zoom_size = 50
            
            # Progress callback to update UI
            def progress_callback(i, total):
                progress = int(20 + (i / total) * 60)  # 20-80% range
                self.detection_progress.setValue(progress)
            
            # Create tubular data
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
            napari.utils.notifications.show_info("Running spine detection...")
            
            # Run the detection algorithm with proper parameters
            spine_positions, dendrite_path, frame_data = detect_spines_with_watershed(
                tube_data=tube_data,
                shaft_radius=params['shaft_radius'],
                min_spine_intensity=params['min_spine_intensity'],
                min_spine_area=params['min_spine_area'],
                min_spine_score=params['min_spine_score'],
                max_spine_distance=params['max_spine_distance'],
                distance_threshold=params['distance_threshold'],
                filter_dendrites=params['filter_dendrites']
            )
            
            self.detection_progress.setValue(90)
            
            # Process the results
            if spine_positions is not None and len(spine_positions) > 0:
                # Create a points layer for the spines
                spine_layer_name = f"Spines - {path_name}"
                
                # Remove existing layer if it exists
                for layer in list(self.viewer.layers):
                    if layer.name == spine_layer_name:
                        self.viewer.layers.remove(layer)
                        break
                
                # Convert spine positions to numpy array
                spine_positions_array = np.array(spine_positions)
                
                # Add the new spine layer
                spine_layer = self.viewer.add_points(
                    spine_positions_array,
                    name=spine_layer_name,
                    size=8,
                    face_color='red',
                    opacity=0.8,
                    symbol='cross'
                )
                
                # Store spine layer reference
                self.state['spine_layers'][path_id] = spine_layer
                
                # Update UI
                self.results_label.setText(f"Results: Detected {len(spine_positions)} spines for {path_name}")
                self.status_label.setText(f"Status: Spine detection completed for {path_name}")
                
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