import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel
)

from path_tracing_module import PathTracingWidget
from dummy_code.segmentation_module_old import SegmentationWidget
from spine_detection_module import SpineDetectionWidget
from visualization_module import PathVisualizationWidget


class NeuroSAMWidget(QWidget):
    """Main widget for the NeuroSAM napari plugin, integrating path tracing, 
    segmentation, and spine detection functionality.
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
        }
        
        # Initialize the waypoints layer
        self.state['waypoints_layer'] = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='Waypoints',
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
        
        # Initialize modules
        self.path_tracing_widget = PathTracingWidget(self.viewer, self.image, self.state)
        self.segmentation_widget = SegmentationWidget(self.viewer, self.image, self.state)
        self.spine_detection_widget = SpineDetectionWidget(self.viewer, self.image, self.state)
        self.path_visualization_widget = PathVisualizationWidget(self.viewer, self.image, self.state)
        
        # Add modules to tabs
        self.tabs.addTab(self.path_tracing_widget, "Path Tracing")
        self.tabs.addTab(self.path_visualization_widget, "Path Management")
        self.tabs.addTab(self.segmentation_widget, "Segmentation")
        self.tabs.addTab(self.spine_detection_widget, "Spine Detection")
        
        # Connect signals between modules
        self._connect_signals()
        
        # Set up event handling for points layers
        self.state['waypoints_layer'].events.data.connect(self.path_tracing_widget.on_waypoints_changed)
        
        # Default mode for waypoints layer
        self.state['waypoints_layer'].mode = 'add'
        
        # Activate the waypoints layer to begin workflow
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Waypoints layer activated. Click on the image to set waypoints.")

    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(3, 3, 3, 3)
        self.setMinimumWidth(300)
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
        self.path_info = QLabel("Path: Not calculated")
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
    
    def on_path_created(self, path_id, path_name, path_data):
        """Handle when a new path is created"""
        self.state['current_path_id'] = path_id
        self.path_info.setText(f"Path: {path_name} with {len(path_data)} points")
        
        # Update all modules with the new path
        self.path_visualization_widget.update_path_list()
        self.segmentation_widget.update_path_list()
        self.spine_detection_widget.update_path_list()
    
    def on_path_updated(self, path_id, path_name, path_data):
        """Handle when a path is updated"""
        self.state['current_path_id'] = path_id
        self.path_info.setText(f"Path: {path_name} with {len(path_data)} points")
        
        # Update visualization
        self.path_visualization_widget.update_path_visualization()
    
    def on_path_selected(self, path_id):
        """Handle when a path is selected from the list"""
        self.state['current_path_id'] = path_id
        path_data = self.state['paths'][path_id]
        self.path_info.setText(f"Path: {path_data['name']} with {len(path_data['data'])} points")
        
        # Update waypoints display
        self.path_tracing_widget.load_path_waypoints(path_id)
    
    def on_path_deleted(self, path_id):
        """Handle when a path is deleted"""
        if not self.state['paths']:
            self.path_info.setText("Path: Not calculated")
            self.state['current_path_id'] = None
        else:
            # Select first available path
            first_path_id = next(iter(self.state['paths']))
            self.on_path_selected(first_path_id)
    
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