import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel
)

from path_tracing_module import PathTracingWidget  # Updated with fast waypoint A*
from segmentation_module import SegmentationWidget
from spine_detection_module import SpineDetectionWidget  # Updated with optimized tube data
from spine_segmentation_module import SpineSegmentationWidget
from visualization_module import PathVisualizationWidget


class NeuroSAMWidget(QWidget):
    """Main widget for the NeuroSAM napari plugin with fast waypoint A* and optimized tube data generation."""
    
    def __init__(self, viewer, image):
        """Initialize the main widget WITHOUT Z-scaling that breaks coordinates"""
        super().__init__()
        self.viewer = viewer
        self.image = image
        
        # Store state shared between modules FIRST
        self.state = {
            'paths': {},              # Dictionary of path data
            'path_layers': {},        # Dictionary of path layers
            'current_path_id': None,  # ID of the currently selected path
            'waypoints_layer': None,  # Layer for waypoints
            'segmentation_layer': None, # Layer for segmentation
            'traced_path_layer': None,  # Layer for traced path visualization
            'spine_positions': [],    # List of detected spine positions
            'spine_layers': {},       # Dictionary of spine layers
            'spine_data': {},         # Enhanced spine detection data
            'spine_segmentation_layers': {},  # Dictionary of spine segmentation layers
            'xy_spacing_nm': 94.0,    # Global XY pixel spacing in nm/pixel (default)
            'z_spacing_nm': 500.0,    # Global Z slice spacing in nm/slice (default)
        }
        
        # CRITICAL FIX: Add image layer WITHOUT any scaling to preserve frame count
        self.image_layer = self.viewer.add_image(
            self.image, 
            name='Image', 
            colormap='gray'
            # NO SCALE parameter - this prevents frame interpolation
        )
        
        print(f"FRAME COUNT CHECK:")
        print(f"Original image shape: {self.image.shape}")
        print(f"Image layer data shape: {self.image_layer.data.shape}")
        print(f"Viewer dims range: {self.viewer.dims.range}")
        
        # Verify frame count is preserved
        if self.image.shape[0] != self.image_layer.data.shape[0]:
            raise ValueError(f"FRAME COUNT MISMATCH! Original: {self.image.shape[0]}, Napari: {self.image_layer.data.shape[0]}")
        
        # Setup UI
        self.setup_ui()
        
        # Initialize the waypoints layer WITHOUT scaling
        self.state['waypoints_layer'] = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='Point Selection',
            size=15,
            face_color='cyan',
            symbol='x'
            # NO SCALE - keep in original pixel coordinates
        )
        
        # Initialize 3D traced path layer WITHOUT scaling
        if self.image.ndim > 2:
            self.state['traced_path_layer'] = self.viewer.add_points(
                np.empty((0, self.image.ndim)),
                name='Traced Path (3D)',
                size=4,
                face_color='magenta',
                opacity=0.7,
                visible=False
                # NO SCALE - keep in original pixel coordinates
            )
        
        # Initialize modules
        self.path_tracing_widget = PathTracingWidget(self.viewer, self.image, self.state)
        self.segmentation_widget = SegmentationWidget(self.viewer, self.image, self.state)
        self.spine_detection_widget = SpineDetectionWidget(self.viewer, self.image, self.state)
        self.spine_segmentation_widget = SpineSegmentationWidget(self.viewer, self.image, self.state)
        self.path_visualization_widget = PathVisualizationWidget(self.viewer, self.image, self.state)
        
        # Add modules to tabs
        self.tabs.addTab(self.path_tracing_widget, "Fast Path Tracing")
        self.tabs.addTab(self.path_visualization_widget, "Path Management")
        self.tabs.addTab(self.segmentation_widget, "Segmentation")
        self.tabs.addTab(self.spine_detection_widget, "Optimized Spine Detection")
        self.tabs.addTab(self.spine_segmentation_widget, "Spine Segmentation")
        
        # Connect signals between modules
        self._connect_signals()
        
        # Connect spacing changes
        self.path_tracing_widget.xy_spacing_spin.valueChanged.connect(self.on_xy_spacing_changed)
        self.path_tracing_widget.z_spacing_spin.valueChanged.connect(self.on_z_spacing_changed)
        
        # Set up event handling for points layers
        self.state['waypoints_layer'].events.data.connect(self.path_tracing_widget.on_waypoints_changed)
        
        # Default mode for waypoints layer
        self.state['waypoints_layer'].mode = 'add'
        
        # Activate the waypoints layer to begin workflow
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Path Tracing ready. Frame count preserved - click points on the dendrite structure.")



    def on_xy_spacing_changed(self, new_xy_spacing):
        """Handle when XY spacing is changed - NO MORE SCALING UPDATES"""
        self.state['xy_spacing_nm'] = new_xy_spacing
        
        # Update all modules with new XY spacing
        self.segmentation_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        self.spine_detection_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        self.spine_segmentation_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        
        print(f"Updated XY spacing to {new_xy_spacing:.1f} nm/pixel")
        print(f"NOTE: No layer scaling applied - coordinates preserved")
    
    def on_z_spacing_changed(self, new_z_spacing):
        """Handle when Z spacing is changed - NO MORE SCALING UPDATES"""
        self.state['z_spacing_nm'] = new_z_spacing
        
        # Update all modules with new Z spacing
        self.segmentation_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        self.spine_detection_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        self.spine_segmentation_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        
        print(f"Updated Z spacing to {new_z_spacing:.1f} nm/slice")
        print(f"NOTE: No layer scaling applied - coordinates preserved")

        
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(3, 3, 3, 3)
        self.setMinimumWidth(320)
        self.setLayout(layout)
        
        # Title
        title = QLabel("<b>Neuro-SAM with Fast Algorithms & Parallel Processing</b>")
        layout.addWidget(title)
        
        # Create tabs for different functionality
        self.tabs = QTabWidget()
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setStyleSheet("QTabBar::tab { height: 22px; }")
        layout.addWidget(self.tabs)
        
        # Current path info at the bottom
        self.path_info = QLabel("Path: Ready for fast tracing")
        layout.addWidget(self.path_info)
    
    def _connect_signals(self):
        """Connect signals between modules for coordination"""
        # Connect path tracing signals
        self.path_tracing_widget.path_created.connect(self.on_path_created)
        self.path_tracing_widget.path_updated.connect(self.on_path_updated)
        
        # Connect path visualization signals
        self.path_visualization_widget.path_selected.connect(self.on_path_selected)
        self.path_visualization_widget.path_deleted.connect(self.on_path_deleted)
        
        # Connect segmentation signals
        self.segmentation_widget.segmentation_completed.connect(self.on_segmentation_completed)
        
        # Connect spine detection signals
        self.spine_detection_widget.spines_detected.connect(self.on_spines_detected)
        
        # Connect spine segmentation signals
        self.spine_segmentation_widget.spine_segmentation_completed.connect(self.on_spine_segmentation_completed)
    
    def on_path_created(self, path_id, path_name, path_data):
        """Handle when a new path is created (including connected paths)"""
        self.state['current_path_id'] = path_id
        
        # Get path information including algorithm and processing details
        path_info = self.state['paths'][path_id]
        num_points = len(path_data)
        
        # Determine algorithm info
        algorithm_info = ""
        if path_info.get('algorithm') == 'fast_waypoint_astar':
            algorithm_info = " (fast algorithm"
            if path_info.get('parallel_processing', False):
                algorithm_info += ", parallel"
            algorithm_info += ")"
        
        # Check if smoothed
        smoothed = path_info.get('smoothed', False)
        smoothing_info = " (smoothed)" if smoothed else ""
        
        # Check if this is a connected path (no original_clicks)
        is_connected = 'original_clicks' in path_info and len(path_info['original_clicks']) == 0
        connected_info = " (connected)" if is_connected else ""
        
        # Create comprehensive status message
        if is_connected:
            message = f"{path_name}: {num_points} points{connected_info}"
        else:
            message = f"{path_name}: {num_points} points{algorithm_info}{smoothing_info}"
        
        self.path_info.setText(f"Path: {message}")
        
        # Update all modules with the new path
        self.path_visualization_widget.update_path_list()
        self.segmentation_widget.update_path_list()
        self.spine_detection_widget.update_path_list()
        self.spine_segmentation_widget.update_path_list()
        
        # Success notification with comprehensive info
        if is_connected:
            napari.utils.notifications.show_info(f"Connected path created! {num_points} points")
        else:
            success_msg = f"Path created! {num_points} points"
            if algorithm_info:
                success_msg += algorithm_info
            if smoothing_info:
                success_msg += smoothing_info
            napari.utils.notifications.show_info(success_msg)
    
    def on_path_updated(self, path_id, path_name, path_data):
        """Handle when a path is updated"""
        self.state['current_path_id'] = path_id
        
        # Get path information including algorithm details
        path_info = self.state['paths'][path_id]
        
        # Build status message
        status_parts = [f"{path_name} with {len(path_data)} points"]
        
        if path_info.get('algorithm') == 'fast_waypoint_astar':
            status_parts.append("(fast algorithm")
            if path_info.get('parallel_processing', False):
                status_parts.append(", parallel")
            status_parts.append(")")
        
        if path_info.get('smoothed', False):
            status_parts.append("(smoothed)")
        
        status_parts.append("(updated)")
        
        status_msg = " ".join(status_parts)
        self.path_info.setText(f"Path: {status_msg}")
        
        # Update visualization
        self.path_visualization_widget.update_path_visualization()
    
    def on_path_selected(self, path_id):
        """Handle when a path is selected from the list"""
        self.state['current_path_id'] = path_id
        path_data = self.state['paths'][path_id]
        
        # Create comprehensive status message
        status_parts = [f"{path_data['name']} with {len(path_data['data'])} points"]
        
        # Add algorithm info
        if path_data.get('algorithm') == 'fast_waypoint_astar':
            status_parts.append("(fast algorithm")
            if path_data.get('parallel_processing', False):
                status_parts.append(", parallel")
            status_parts.append(")")
        
        # Add other attributes
        if path_data.get('smoothed', False):
            status_parts.append("(smoothed)")
        
        is_connected = 'original_clicks' in path_data and len(path_data['original_clicks']) == 0
        if is_connected:
            status_parts.append("(connected)")
        
        message = " ".join(status_parts)
        self.path_info.setText(f"Path: {message}")
        
        # Update waypoints display
        self.path_tracing_widget.load_path_waypoints(path_id)
    
    def on_path_deleted(self, path_id):
        """Handle when a path is deleted"""
        if not self.state['paths']:
            self.path_info.setText("Path: Ready for fast tracing")
            self.state['current_path_id'] = None
        else:
            # Select first available path
            first_path_id = next(iter(self.state['paths']))
            self.on_path_selected(first_path_id)
        
        # Update all modules after path deletion
        self.segmentation_widget.update_path_list()
        self.spine_detection_widget.update_path_list()
        self.spine_segmentation_widget.update_path_list()
    
    def on_segmentation_completed(self, path_id, layer_name):
        """Handle when segmentation is completed for a path"""
        path_data = self.state['paths'][path_id]
        
        # Build comprehensive status message
        status_parts = [f"Segmentation completed for {path_data['name']}"]
        
        if path_data.get('algorithm') == 'fast_waypoint_astar':
            status_parts.append("(fast algorithm path)")
        elif path_data.get('smoothed', False):
            status_parts.append("(smoothed path)")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Enable spine detection for this path
        self.spine_detection_widget.enable_for_path(path_id)
    
    def on_spines_detected(self, path_id, spine_positions):
        """Handle when spines are detected for a path"""
        path_data = self.state['paths'][path_id]
        
        # Build comprehensive status message
        status_parts = [f"Detected {len(spine_positions)} spines for {path_data['name']}"]
        
        # Add processing method info
        if 'spine_data' in self.state and path_id in self.state['spine_data']:
            spine_info = self.state['spine_data'][path_id]
            if spine_info.get('detection_method') == 'optimized_angle_based_extended':
                if spine_info.get('parameters', {}).get('enable_parallel', False):
                    status_parts.append("(optimized, parallel)")
                else:
                    status_parts.append("(optimized)")
        
        # Add path algorithm info
        if path_data.get('algorithm') == 'fast_waypoint_astar':
            status_parts.append("(fast algorithm path)")
        elif path_data.get('smoothed', False):
            status_parts.append("(smoothed path)")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Store spine positions in state
        self.state['spine_positions'] = spine_positions
        
        # Enable spine segmentation for this path
        self.spine_segmentation_widget.enable_for_path(path_id)
    
    def on_spine_segmentation_completed(self, path_id, layer_name):
        """Handle when spine segmentation is completed for a path"""
        path_data = self.state['paths'][path_id]
        
        # Build comprehensive completion message
        status_parts = [f"Spine segmentation completed for {path_data['name']}"]
        
        # Add algorithm details
        algorithm_details = []
        if path_data.get('algorithm') == 'fast_waypoint_astar':
            algorithm_details.append("fast path tracing")
        
        if 'spine_data' in self.state and path_id in self.state['spine_data']:
            spine_info = self.state['spine_data'][path_id]
            if spine_info.get('detection_method') == 'optimized_angle_based_extended':
                algorithm_details.append("optimized spine detection")
        
        if algorithm_details:
            status_parts.append(f"({', '.join(algorithm_details)})")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Create comprehensive completion notification
        notification_parts = [f"Complete workflow finished for {path_data['name']}"]
        if algorithm_details:
            notification_parts.append(f"using {', '.join(algorithm_details)}")
        
        napari.utils.notifications.show_info(" ".join(notification_parts))
    
    def on_xy_spacing_changed(self, new_xy_spacing):
        """Handle when XY spacing is changed"""
        self.state['xy_spacing_nm'] = new_xy_spacing
        
        # Update 3D view scaling
        self._update_image_scaling()
        
        # Update all modules with new XY spacing
        self.segmentation_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        self.spine_detection_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        self.spine_segmentation_widget.update_spacing(new_xy_spacing, self.state['z_spacing_nm'])
        
        print(f"Updated XY spacing to {new_xy_spacing:.1f} nm/pixel")
    
    def on_z_spacing_changed(self, new_z_spacing):
        """Handle when Z spacing is changed"""
        self.state['z_spacing_nm'] = new_z_spacing
        
        # Update 3D view scaling
        self._update_image_scaling()
        
        # Update all modules with new Z spacing
        self.segmentation_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        self.spine_detection_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        self.spine_segmentation_widget.update_spacing(self.state['xy_spacing_nm'], new_z_spacing)
        
        print(f"Updated Z spacing to {new_z_spacing:.1f} nm/slice")
    
    def _update_image_scaling(self):
        """Update the 3D view scaling based on current spacing"""
        z_scale = self.state['z_spacing_nm'] / self.state['xy_spacing_nm']
        
        # Update ALL layers with the same scaling to maintain coordinate consistency
        self.image_layer.scale = (z_scale, 1.0, 1.0)
        
        if self.state['waypoints_layer'] is not None:
            self.state['waypoints_layer'].scale = (z_scale, 1.0, 1.0)
        
        if self.state['traced_path_layer'] is not None:
            self.state['traced_path_layer'].scale = (z_scale, 1.0, 1.0)
        
        # Update any existing path layers
        for layer in self.state['path_layers'].values():
            if layer is not None:
                layer.scale = (z_scale, 1.0, 1.0)
        
        # Update any existing spine layers
        for layer in self.state.get('spine_layers', {}).values():
            if layer is not None:
                layer.scale = (z_scale, 1.0, 1.0)
        
        # Update segmentation layer if it exists
        if self.state.get('segmentation_layer') is not None:
            self.state['segmentation_layer'].scale = (z_scale, 1.0, 1.0)
        
        # Update spine segmentation layers if they exist
        for layer in self.state.get('spine_segmentation_layers', {}).values():
            if layer is not None:
                layer.scale = (z_scale, 1.0, 1.0)
        
        print(f"Updated 3D view scaling: Z/XY ratio = {z_scale:.2f}")
        print(f"Applied scaling (z_scale, 1.0, 1.0) = ({z_scale:.2f}, 1.0, 1.0) to all layers")