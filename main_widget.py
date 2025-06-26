import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel
)

from path_tracing_module import PathTracingWidget  # Back to original name
from segmentation_module import SegmentationWidget
from spine_detection_module import SpineDetectionWidget
from spine_segmentation_module import SpineSegmentationWidget
from visualization_module import PathVisualizationWidget


class NeuroSAMWidget(QWidget):
    """Main widget for the NeuroSAM napari plugin, integrating path tracing, 
    segmentation, spine detection, and spine segmentation functionality.
    """
    
    def __init__(self, viewer, image):
        """Initialize the main widget.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance
        image : numpy.ndarray
            3D or higher-dimensional image data
        """
        super().__init__()
        self.viewer = viewer
        self.image = image
        
        # Initialize the image layer
        self.image_layer = self.viewer.add_image(
            self.image, name='Image', colormap='gray'
        )
        
        # Setup UI
        self.setup_ui()
        
        # Store state shared between modules
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
        }
        
        # Initialize the waypoints layer
        self.state['waypoints_layer'] = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='Point Selection',
            size=15,
            face_color='cyan',
            symbol='x'
        )
        
        # Initialize 3D traced path layer if applicable
        if self.image.ndim > 2:
            self.state['traced_path_layer'] = self.viewer.add_points(
                np.empty((0, self.image.ndim)),
                name='Traced Path (3D)',
                size=4,
                face_color='magenta',
                opacity=0.7,
                visible=False
            )
        
        # Initialize modules (original interface, enhanced backend)
        self.path_tracing_widget = PathTracingWidget(self.viewer, self.image, self.state)
        self.segmentation_widget = SegmentationWidget(self.viewer, self.image, self.state)
        self.spine_detection_widget = SpineDetectionWidget(self.viewer, self.image, self.state)
        self.spine_segmentation_widget = SpineSegmentationWidget(self.viewer, self.image, self.state)
        self.path_visualization_widget = PathVisualizationWidget(self.viewer, self.image, self.state)
        
        # Add modules to tabs
        self.tabs.addTab(self.path_tracing_widget, "Path Tracing")
        self.tabs.addTab(self.path_visualization_widget, "Path Management")
        self.tabs.addTab(self.segmentation_widget, "Segmentation")
        self.tabs.addTab(self.spine_detection_widget, "Spine Detection")
        self.tabs.addTab(self.spine_segmentation_widget, "Spine Segmentation")
        
        # Connect signals between modules
        self._connect_signals()
        
        # Set up event handling for points layers
        self.state['waypoints_layer'].events.data.connect(self.path_tracing_widget.on_waypoints_changed)
        
        # Default mode for waypoints layer
        self.state['waypoints_layer'].mode = 'add'
        
        # Activate the waypoints layer to begin workflow
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Path Tracing ready. Click points on the dendrite structure.")

    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(3, 3, 3, 3)
        self.setMinimumWidth(320)  # Original width
        self.setLayout(layout)
        
        # Title
        title = QLabel("<b>Neuro-SAM</b>")
        layout.addWidget(title)
        
        # Create tabs for different functionality
        self.tabs = QTabWidget()
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setStyleSheet("QTabBar::tab { height: 22px; }")
        layout.addWidget(self.tabs)
        
        # Current path info at the bottom
        self.path_info = QLabel("Path: Ready for tracing")
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
        """Handle when a new path is created"""
        self.state['current_path_id'] = path_id
        
        # Simple status message (no enhanced details in UI)
        num_points = len(path_data)
        message = f"{path_name}: {num_points} points"
        self.path_info.setText(f"Path: {message}")
        
        # Update all modules with the new path
        self.path_visualization_widget.update_path_list()
        self.segmentation_widget.update_path_list()
        self.spine_detection_widget.update_path_list()
        self.spine_segmentation_widget.update_path_list()
        
        # Simple success notification
        napari.utils.notifications.show_info(f"Path created successfully! {num_points} points")
    
    def on_path_updated(self, path_id, path_name, path_data):
        """Handle when a path is updated"""
        self.state['current_path_id'] = path_id
        self.path_info.setText(f"Path: {path_name} with {len(path_data)} points (Updated)")
        
        # Update visualization
        self.path_visualization_widget.update_path_visualization()
    
    def on_path_selected(self, path_id):
        """Handle when a path is selected from the list"""
        self.state['current_path_id'] = path_id
        path_data = self.state['paths'][path_id]
        
        # Simple status message
        message = f"{path_data['name']} with {len(path_data['data'])} points"
        self.path_info.setText(f"Path: {message}")
        
        # Update waypoints display
        self.path_tracing_widget.load_path_waypoints(path_id)
    
    def on_path_deleted(self, path_id):
        """Handle when a path is deleted"""
        if not self.state['paths']:
            self.path_info.setText("Path: Ready for tracing")
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
        self.path_info.setText(f"Segmentation completed for {path_data['name']}")
        
        # Enable spine detection for this path
        self.spine_detection_widget.enable_for_path(path_id)
    
    def on_spines_detected(self, path_id, spine_positions):
        """Handle when spines are detected for a path"""
        path_data = self.state['paths'][path_id]
        self.path_info.setText(f"Detected {len(spine_positions)} spines for {path_data['name']}")
        
        # Store spine positions in state
        self.state['spine_positions'] = spine_positions
        
        # Enable spine segmentation for this path
        self.spine_segmentation_widget.enable_for_path(path_id)
    
    def on_spine_segmentation_completed(self, path_id, layer_name):
        """Handle when spine segmentation is completed for a path"""
        path_data = self.state['paths'][path_id]
        self.path_info.setText(f"Spine segmentation completed for {path_data['name']}")
        
        napari.utils.notifications.show_info(f"Complete workflow finished for {path_data['name']}")