import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QPushButton, QHBoxLayout
)

from path_tracing_module import PathTracingWidget
from segmentation_module import SegmentationWidget
from spine_detection_module import SpineDetectionWidget
from spine_segmentation_module import SpineSegmentationWidget
from visualization_module import PathVisualizationWidget
# Import the new scaling system
from anisotropic_scaling import AnisotropicScaler


class NeuroSAMWidget(QWidget):
    """Main widget for the NeuroSAM napari plugin with anisotropic scaling support."""
    
    def __init__(self, viewer, image, original_spacing_xyz=(94.0, 94.0, 500.0)):
        """Initialize the main widget with anisotropic scaling.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance
        image : numpy.ndarray
            3D or higher-dimensional image data
        original_spacing_xyz : tuple
            Original voxel spacing in (x, y, z) nanometers
        """
        super().__init__()
        self.viewer = viewer
        self.original_image = image
        self.current_image = image.copy()
        
        # Initialize anisotropic scaler
        self.scaler = AnisotropicScaler(original_spacing_xyz)
        
        # Initialize the image layer with original image
        self.image_layer = self.viewer.add_image(
            self.current_image, 
            name=f'Image (spacing: {original_spacing_xyz[0]:.1f}, {original_spacing_xyz[1]:.1f}, {original_spacing_xyz[2]:.1f} nm)', 
            colormap='gray'
        )
        
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
            'current_spacing_xyz': original_spacing_xyz,  # Current voxel spacing
            'scaler': self.scaler,    # Reference to scaler for coordinate conversion
        }
        
        # Initialize the waypoints layer
        self.state['waypoints_layer'] = self.viewer.add_points(
            np.empty((0, self.current_image.ndim)),
            name='Point Selection',
            size=15,
            face_color='cyan',
            symbol='x'
        )
        
        # Initialize 3D traced path layer if applicable
        if self.current_image.ndim > 2:
            self.state['traced_path_layer'] = self.viewer.add_points(
                np.empty((0, self.current_image.ndim)),
                name='Traced Path (3D)',
                size=4,
                face_color='magenta',
                opacity=0.7,
                visible=False
            )
        
        # Initialize modules with scaled image support
        self.path_tracing_widget = PathTracingWidget(
            self.viewer, self.current_image, self.state, self.scaler, self._on_scaling_update
        )
        self.segmentation_widget = SegmentationWidget(self.viewer, self.current_image, self.state)
        self.spine_detection_widget = SpineDetectionWidget(self.viewer, self.current_image, self.state)
        self.spine_segmentation_widget = SpineSegmentationWidget(self.viewer, self.current_image, self.state)
        self.path_visualization_widget = PathVisualizationWidget(self.viewer, self.current_image, self.state)
        
        # Setup UI
        self.setup_ui()
        
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
        napari.utils.notifications.show_info("NeuroSAM ready. Configure scaling in Path Tracing tab, then start analysis.")

    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(3, 3, 3, 3)
        self.setMinimumWidth(320)
        self.setLayout(layout)
        
        # Title
        title = QLabel("<b>Neuro-SAM with Anisotropic Scaling</b>")
        layout.addWidget(title)
        
        # Create tabs for different functionality
        self.tabs = QTabWidget()
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setStyleSheet("QTabBar::tab { height: 22px; }")
        layout.addWidget(self.tabs)
        
        # Add export button (removed - will use individual module export buttons)
        # export_layout = QHBoxLayout()
        # self.export_all_btn = QPushButton("Export All at Original Scale")
        # self.export_all_btn.setFixedHeight(22)
        # self.export_all_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        # self.export_all_btn.clicked.connect(self.export_analysis_results)
        # self.export_all_btn.setEnabled(False)
        # self.export_all_btn.setToolTip("Export all paths and masks rescaled back to original image dimensions")
        # export_layout.addWidget(self.export_all_btn)
        # layout.addLayout(export_layout)
        
        # Current path info at the bottom
        self.path_info = QLabel("Status: Ready for analysis")
        layout.addWidget(self.path_info)
    
    def _on_scaling_update(self, interpolation_order):
        """
        Handle when scaling is updated
        
        Args:
            interpolation_order: Interpolation order for scaling
        """
        try:
            # Store old image shape and spacing for coordinate conversion
            old_image_shape = self.current_image.shape
            old_spacing = self.scaler.get_effective_spacing()
            
            # Scale the image
            scaled_image = self.scaler.scale_image(
                self.original_image, 
                order=interpolation_order
            )
            
            # Calculate the actual transformation between old and new image
            new_image_shape = scaled_image.shape
            
            # This is the key: calculate the direct coordinate transformation
            coordinate_scale_factors = np.array(new_image_shape) / np.array(old_image_shape)
            
            print(f"Old image shape: {old_image_shape}")
            print(f"New image shape: {new_image_shape}")
            print(f"Coordinate scale factors: {coordinate_scale_factors}")
            
            # Update current image
            self.current_image = scaled_image
            
            # Update the napari layer
            spacing_str = f"{self.scaler.current_spacing_xyz[0]:.1f}, {self.scaler.current_spacing_xyz[1]:.1f}, {self.scaler.current_spacing_xyz[2]:.1f}"
            self.image_layer.data = scaled_image
            self.image_layer.name = f"Image (spacing: {spacing_str} nm)"
            
            # Update state
            new_spacing = self.scaler.get_effective_spacing()
            self.state['current_spacing_xyz'] = new_spacing
            
            # Scale existing analysis using the direct coordinate transformation
            self._transform_analysis_coordinates(coordinate_scale_factors, new_image_shape)
            
            # Update all modules with new image and spacing
            self._update_modules_with_scaled_image()
            
            # Update status
            scale_factors = self.scaler.get_scale_factors()
            self.path_info.setText(
                f"Status: Scaled to {spacing_str} nm "
                f"(factors: Z={scale_factors[0]:.2f}, Y={scale_factors[1]:.2f}, X={scale_factors[2]:.2f})"
            )
            
            # Count preserved analysis (no need to enable export button anymore)
            num_paths = len(self.state['paths'])
            num_segmentations = len([layer for layer in self.viewer.layers if 'Segmentation -' in layer.name])
            num_spines = len([layer for layer in self.viewer.layers if 'Spine' in layer.name])
            
            if num_paths > 0 or num_segmentations > 0 or num_spines > 0:
                napari.utils.notifications.show_info(
                    f"Scaled to {spacing_str} nm. Preserved: {num_paths} paths, "
                    f"{num_segmentations} segmentations, {num_spines} spine layers"
                )
            else:
                napari.utils.notifications.show_info(f"Image scaled successfully to {spacing_str} nm")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating scaled image: {str(e)}")
            print(f"Scaling update error: {str(e)}")
    
    def _transform_analysis_coordinates(self, coordinate_scale_factors, new_image_shape):
        """
        Transform all analysis coordinates using direct coordinate transformation
        This ensures paths and masks move with the image when it's scaled
        
        Args:
            coordinate_scale_factors: Direct transformation factors [Z, Y, X]
            new_image_shape: Shape of the new scaled image
        """
        try:
            print(f"Transforming analysis coordinates with factors: {coordinate_scale_factors}")
            
            # Transform all existing paths
            for path_id, path_data in self.state['paths'].items():
                old_path_coords = path_data['data']
                
                # Apply direct coordinate transformation
                new_path_coords = old_path_coords * coordinate_scale_factors[np.newaxis, :]
                path_data['data'] = new_path_coords
                
                # Update the path layer immediately
                if path_id in self.state['path_layers']:
                    layer = self.state['path_layers'][path_id]
                    layer.data = new_path_coords
                    print(f"Transformed path {path_data['name']}")
                    print(f"  Old range: Z[{old_path_coords[:,0].min():.1f}-{old_path_coords[:,0].max():.1f}]")
                    print(f"  New range: Z[{new_path_coords[:,0].min():.1f}-{new_path_coords[:,0].max():.1f}]")
                
                # Transform other coordinate data
                if 'start' in path_data and path_data['start'] is not None:
                    path_data['start'] = path_data['start'] * coordinate_scale_factors
                
                if 'end' in path_data and path_data['end'] is not None:
                    path_data['end'] = path_data['end'] * coordinate_scale_factors
                
                if 'waypoints' in path_data and path_data['waypoints']:
                    scaled_waypoints = []
                    for waypoint in path_data['waypoints']:
                        scaled_waypoint = waypoint * coordinate_scale_factors
                        scaled_waypoints.append(scaled_waypoint)
                    path_data['waypoints'] = scaled_waypoints
                
                if 'original_clicks' in path_data and path_data['original_clicks']:
                    scaled_clicks = []
                    for click in path_data['original_clicks']:
                        scaled_click = click * coordinate_scale_factors
                        scaled_clicks.append(scaled_click)
                    path_data['original_clicks'] = scaled_clicks
            
            # Transform waypoints layer
            if self.state['waypoints_layer'] is not None and len(self.state['waypoints_layer'].data) > 0:
                old_waypoints = self.state['waypoints_layer'].data
                new_waypoints = old_waypoints * coordinate_scale_factors[np.newaxis, :]
                self.state['waypoints_layer'].data = new_waypoints
                print(f"Transformed waypoints layer")
            
            # Transform segmentation masks to match new image dimensions
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Segmentation -' in layer.name:
                    old_mask = layer.data
                    
                    # Use scipy zoom to transform mask to new dimensions
                    from scipy.ndimage import zoom
                    zoom_factors = np.array(new_image_shape) / np.array(old_mask.shape)
                    new_mask = zoom(old_mask, zoom_factors, order=0, prefilter=False)
                    
                    # Ensure binary values
                    new_mask = (new_mask > 0.5).astype(old_mask.dtype)
                    
                    layer.data = new_mask
                    print(f"Transformed segmentation mask {layer.name}: {old_mask.shape} -> {new_mask.shape}")
            
            # Transform spine segmentation masks
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Spine Segmentation -' in layer.name:
                    old_mask = layer.data
                    
                    # Use scipy zoom to transform mask to new dimensions
                    from scipy.ndimage import zoom
                    zoom_factors = np.array(new_image_shape) / np.array(old_mask.shape)
                    new_mask = zoom(old_mask, zoom_factors, order=0, prefilter=False)
                    
                    # Ensure binary values
                    new_mask = (new_mask > 0.5).astype(old_mask.dtype)
                    
                    layer.data = new_mask
                    print(f"Transformed spine segmentation mask {layer.name}: {old_mask.shape} -> {new_mask.shape}")
            
            # Transform spine positions
            for path_id, spine_layer in self.state.get('spine_layers', {}).items():
                if len(spine_layer.data) > 0:
                    old_spine_coords = spine_layer.data
                    new_spine_coords = old_spine_coords * coordinate_scale_factors[np.newaxis, :]
                    spine_layer.data = new_spine_coords
                    print(f"Transformed spine positions for path {path_id}")
            
            # Transform spine data
            if 'spine_data' in self.state:
                for path_id, spine_info in self.state['spine_data'].items():
                    if 'original_positions' in spine_info:
                        old_positions = spine_info['original_positions']
                        new_positions = old_positions * coordinate_scale_factors[np.newaxis, :]
                        spine_info['original_positions'] = new_positions
            
            # Transform spine_positions in state
            if self.state.get('spine_positions') is not None and len(self.state['spine_positions']) > 0:
                old_spine_positions = self.state['spine_positions']
                new_spine_positions = old_spine_positions * coordinate_scale_factors[np.newaxis, :]
                self.state['spine_positions'] = new_spine_positions
            
            # Transform traced path layer
            if (self.state.get('traced_path_layer') is not None and 
                len(self.state['traced_path_layer'].data) > 0):
                old_traced = self.state['traced_path_layer'].data
                new_traced = old_traced * coordinate_scale_factors[np.newaxis, :]
                self.state['traced_path_layer'].data = new_traced
                print(f"Transformed traced path layer")
            
            print(f"Successfully transformed all analysis to new image dimensions: {new_image_shape}")
            
        except Exception as e:
            print(f"Error transforming analysis coordinates: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _scale_existing_analysis(self, old_spacing_xyz, new_spacing_xyz, new_image_shape):
        """
        Scale existing paths, segmentation masks, and spine data to match new image scaling
        This ensures visual consistency - everything scales together with the image
        
        Args:
            old_spacing_xyz: Previous spacing (x, y, z) in nm
            new_spacing_xyz: New spacing (x, y, z) in nm  
            new_image_shape: Shape of the new scaled image
        """
        try:
            print(f"Scaling existing analysis from {old_spacing_xyz} to {new_spacing_xyz} nm")
            print(f"Old image shape: {self.original_image.shape}")
            print(f"New image shape: {new_image_shape}")
            
            # Calculate the scale factors from the scaler (this is what was applied to the image)
            scale_factors = self.scaler.get_scale_factors()  # [Z, Y, X] order
            print(f"Scale factors (Z,Y,X): {scale_factors}")
            
            # Scale all existing paths using the same scale factors as the image
            for path_id, path_data in self.state['paths'].items():
                old_path_coords = path_data['data']
                
                # Apply the same scaling factors to path coordinates
                # path coordinates are in [Z, Y, X] format, same as scale_factors
                new_path_coords = old_path_coords * scale_factors[np.newaxis, :]
                path_data['data'] = new_path_coords
                
                # Update the path layer to show the scaled coordinates
                if path_id in self.state['path_layers']:
                    layer = self.state['path_layers'][path_id]
                    layer.data = new_path_coords
                    print(f"Updated path layer {path_data['name']} with new coordinates")
                
                # Scale other coordinate data
                if 'start' in path_data and path_data['start'] is not None:
                    path_data['start'] = path_data['start'] * scale_factors
                
                if 'end' in path_data and path_data['end'] is not None:
                    path_data['end'] = path_data['end'] * scale_factors
                
                if 'waypoints' in path_data and path_data['waypoints']:
                    scaled_waypoints = []
                    for waypoint in path_data['waypoints']:
                        scaled_waypoint = waypoint * scale_factors
                        scaled_waypoints.append(scaled_waypoint)
                    path_data['waypoints'] = scaled_waypoints
                
                if 'original_clicks' in path_data and path_data['original_clicks']:
                    scaled_clicks = []
                    for click in path_data['original_clicks']:
                        scaled_click = click * scale_factors
                        scaled_clicks.append(scaled_click)
                    path_data['original_clicks'] = scaled_clicks
                
                # Update spacing metadata
                path_data['voxel_spacing_xyz'] = new_spacing_xyz
                
                print(f"Scaled path {path_data['name']}: shape {old_path_coords.shape} -> {new_path_coords.shape}")
                print(f"  Old coords range: Z[{old_path_coords[:,0].min():.1f}-{old_path_coords[:,0].max():.1f}], "
                      f"Y[{old_path_coords[:,1].min():.1f}-{old_path_coords[:,1].max():.1f}], "
                      f"X[{old_path_coords[:,2].min():.1f}-{old_path_coords[:,2].max():.1f}]")
                print(f"  New coords range: Z[{new_path_coords[:,0].min():.1f}-{new_path_coords[:,0].max():.1f}], "
                      f"Y[{new_path_coords[:,1].min():.1f}-{new_path_coords[:,1].max():.1f}], "
                      f"X[{new_path_coords[:,2].min():.1f}-{new_path_coords[:,2].max():.1f}]")
            
            # Scale waypoints layer using the same scale factors
            if self.state['waypoints_layer'] is not None and len(self.state['waypoints_layer'].data) > 0:
                old_waypoints = self.state['waypoints_layer'].data
                new_waypoints = old_waypoints * scale_factors[np.newaxis, :]
                self.state['waypoints_layer'].data = new_waypoints
                print(f"Scaled waypoints layer: {len(old_waypoints)} points")
            
            # Scale segmentation masks to match new image dimensions
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Segmentation -' in layer.name:
                    old_mask = layer.data
                    print(f"Scaling segmentation mask {layer.name}: {old_mask.shape} -> {new_image_shape}")
                    
                    # Use scipy zoom to scale the mask to match the new image shape
                    from scipy.ndimage import zoom
                    zoom_factors = np.array(new_image_shape) / np.array(old_mask.shape)
                    new_mask = zoom(old_mask, zoom_factors, order=0, prefilter=False)
                    
                    # Ensure binary values
                    new_mask = (new_mask > 0.5).astype(old_mask.dtype)
                    
                    layer.data = new_mask
                    print(f"Scaled segmentation mask {layer.name}: {old_mask.shape} -> {new_mask.shape}")
            
            # Scale spine segmentation masks to match new image dimensions
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Spine Segmentation -' in layer.name:
                    old_mask = layer.data
                    print(f"Scaling spine segmentation mask {layer.name}: {old_mask.shape} -> {new_image_shape}")
                    
                    # Use scipy zoom to scale the mask to match the new image shape
                    from scipy.ndimage import zoom
                    zoom_factors = np.array(new_image_shape) / np.array(old_mask.shape)
                    new_mask = zoom(old_mask, zoom_factors, order=0, prefilter=False)
                    
                    # Ensure binary values
                    new_mask = (new_mask > 0.5).astype(old_mask.dtype)
                    
                    layer.data = new_mask
                    print(f"Scaled spine segmentation mask {layer.name}: {old_mask.shape} -> {new_mask.shape}")
            
            # Scale spine positions using the same scale factors
            for path_id, spine_layer in self.state.get('spine_layers', {}).items():
                if len(spine_layer.data) > 0:
                    old_spine_coords = spine_layer.data
                    new_spine_coords = old_spine_coords * scale_factors[np.newaxis, :]
                    spine_layer.data = new_spine_coords
                    print(f"Scaled spine positions for path {path_id}: {len(old_spine_coords)} positions")
            
            # Scale spine data
            if 'spine_data' in self.state:
                for path_id, spine_info in self.state['spine_data'].items():
                    if 'original_positions' in spine_info:
                        old_positions = spine_info['original_positions']
                        new_positions = old_positions * scale_factors[np.newaxis, :]
                        spine_info['original_positions'] = new_positions
                        spine_info['detection_spacing'] = new_spacing_xyz
            
            # Scale spine_positions in state
            if self.state.get('spine_positions') is not None and len(self.state['spine_positions']) > 0:
                old_spine_positions = self.state['spine_positions']
                new_spine_positions = old_spine_positions * scale_factors[np.newaxis, :]
                self.state['spine_positions'] = new_spine_positions
            
            # Scale traced path layer
            if (self.state.get('traced_path_layer') is not None and 
                len(self.state['traced_path_layer'].data) > 0):
                old_traced = self.state['traced_path_layer'].data
                new_traced = old_traced * scale_factors[np.newaxis, :]
                self.state['traced_path_layer'].data = new_traced
                print(f"Scaled traced path layer: {len(old_traced)} points")
            
            # Force napari to refresh the display
            self.viewer.dims.refresh()
            
            print(f"Successfully scaled all analysis to match new image dimensions: {new_image_shape}")
            napari.utils.notifications.show_info(f"Successfully scaled all analysis to new spacing: {new_spacing_xyz[0]:.1f}, {new_spacing_xyz[1]:.1f}, {new_spacing_xyz[2]:.1f} nm")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error scaling existing analysis: {str(e)}")
            print(f"Error in _scale_existing_analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_modules_with_scaled_image(self):
        """Update all modules with the new scaled image"""
        try:
            # Update each module's image reference
            self.path_tracing_widget.image = self.current_image
            self.segmentation_widget.image = self.current_image
            self.spine_detection_widget.image = self.current_image
            self.spine_segmentation_widget.image = self.current_image
            self.path_visualization_widget.image = self.current_image
            
            # Update spacing information in modules that use it
            current_spacing = self.scaler.get_effective_spacing()
            
            # Update other modules' spacing
            min_spacing = min(current_spacing)
            if hasattr(self.segmentation_widget, 'update_pixel_spacing'):
                self.segmentation_widget.update_pixel_spacing(min_spacing)
            if hasattr(self.spine_detection_widget, 'update_pixel_spacing'):
                self.spine_detection_widget.update_pixel_spacing(min_spacing)
            if hasattr(self.spine_segmentation_widget, 'update_pixel_spacing'):
                self.spine_segmentation_widget.update_pixel_spacing(min_spacing)
            
            # Update path lists in all modules
            self.segmentation_widget.update_path_list()
            self.spine_detection_widget.update_path_list()
            self.spine_segmentation_widget.update_path_list()
            self.path_visualization_widget.update_path_list()
            
        except Exception as e:
            print(f"Error updating modules with scaled image: {str(e)}")
    
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
        
        # Store coordinates in original image space for future reference
        if 'coordinates_original_space' not in path_info:
            original_coords = self.scaler.unscale_coordinates(path_data)
            path_info['coordinates_original_space'] = original_coords
            path_info['scaling_applied'] = self.scaler.get_effective_spacing()
        
        # Create comprehensive status message
        spacing = self.scaler.get_effective_spacing()
        algorithm_info = ""
        if path_info.get('algorithm') == 'waypoint_astar':
            algorithm_info = " (waypoint_astar"
            if path_info.get('parallel_processing', False):
                algorithm_info += ", parallel"
            algorithm_info += ")"
        
        smoothed = path_info.get('smoothed', False)
        smoothing_info = " (smoothed)" if smoothed else ""
        
        is_connected = 'original_clicks' in path_info and len(path_info['original_clicks']) == 0
        connected_info = " (connected)" if is_connected else ""
        
        scaling_info = f" [X={spacing[0]:.1f}, Y={spacing[1]:.1f}, Z={spacing[2]:.1f} nm]"
        
        if is_connected:
            message = f"{path_name}: {num_points} points{connected_info}{scaling_info}"
        else:
            message = f"{path_name}: {num_points} points{algorithm_info}{smoothing_info}{scaling_info}"
        
        self.path_info.setText(f"Path: {message}")
        
        # Update all modules with the new path
        self.path_visualization_widget.update_path_list()
        self.segmentation_widget.update_path_list()
        self.spine_detection_widget.update_path_list()
        self.spine_segmentation_widget.update_path_list()
        
        # Success notification
        if is_connected:
            napari.utils.notifications.show_info(f"Connected path created! {num_points} points at current spacing")
        else:
            success_msg = f"Path created! {num_points} points"
            if algorithm_info:
                success_msg += algorithm_info
            if smoothing_info:
                success_msg += smoothing_info
            success_msg += f" at spacing {spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f} nm"
            napari.utils.notifications.show_info(success_msg)
    
    def on_path_updated(self, path_id, path_name, path_data):
        """Handle when a path is updated"""
        self.state['current_path_id'] = path_id
        
        # Update coordinates in original space
        if path_id in self.state['paths']:
            path_info = self.state['paths'][path_id]
            original_coords = self.scaler.unscale_coordinates(path_data)
            path_info['coordinates_original_space'] = original_coords
            path_info['scaling_applied'] = self.scaler.get_effective_spacing()
        
        # Build status message with scaling info
        spacing = self.scaler.get_effective_spacing()
        path_info = self.state['paths'][path_id]
        
        status_parts = [f"{path_name} with {len(path_data)} points"]
        
        if path_info.get('algorithm') == 'waypoint_astar':
            status_parts.append("(waypoint_astar")
            if path_info.get('parallel_processing', False):
                status_parts.append(", parallel")
            status_parts.append(")")
        
        if path_info.get('smoothed', False):
            status_parts.append("(smoothed)")
        
        status_parts.append("(updated)")
        status_parts.append(f"[X={spacing[0]:.1f}, Y={spacing[1]:.1f}, Z={spacing[2]:.1f} nm]")
        
        status_msg = " ".join(status_parts)
        self.path_info.setText(f"Path: {status_msg}")
        
        # Update visualization
        self.path_visualization_widget.update_path_visualization()
    
    def on_path_selected(self, path_id):
        """Handle when a path is selected from the list"""
        self.state['current_path_id'] = path_id
        path_data = self.state['paths'][path_id]
        
        # Create comprehensive status message with scaling
        spacing = self.scaler.get_effective_spacing()
        status_parts = [f"{path_data['name']} with {len(path_data['data'])} points"]
        
        # Add algorithm info
        if path_data.get('algorithm') == 'waypoint_astar':
            status_parts.append("(waypoint_astar")
            if path_data.get('parallel_processing', False):
                status_parts.append(", parallel")
            status_parts.append(")")
        
        # Add other attributes
        if path_data.get('smoothed', False):
            status_parts.append("(smoothed)")
        
        is_connected = 'original_clicks' in path_data and len(path_data['original_clicks']) == 0
        if is_connected:
            status_parts.append("(connected)")
        
        # Add current scaling info
        status_parts.append(f"[X={spacing[0]:.1f}, Y={spacing[1]:.1f}, Z={spacing[2]:.1f} nm]")
        
        message = " ".join(status_parts)
        self.path_info.setText(f"Path: {message}")
        
        # Update waypoints display
        self.path_tracing_widget.load_path_waypoints(path_id)
    
    def on_path_deleted(self, path_id):
        """Handle when a path is deleted"""
        if not self.state['paths']:
            spacing = self.scaler.get_effective_spacing()
            self.path_info.setText(f"Path: Ready for tracing at {spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f} nm spacing")
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
        spacing = self.scaler.get_effective_spacing()
        
        # Build comprehensive status message
        status_parts = [f"Segmentation completed for {path_data['name']}"]
        
        if path_data.get('algorithm') == 'waypoint_astar':
            status_parts.append("(waypoint_astar path)")
        elif path_data.get('smoothed', False):
            status_parts.append("(smoothed path)")
        
        status_parts.append(f"at {spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f} nm")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Enable spine detection for this path
        self.spine_detection_widget.enable_for_path(path_id)
    
    def on_spines_detected(self, path_id, spine_positions):
        """Handle when spines are detected for a path"""
        path_data = self.state['paths'][path_id]
        spacing = self.scaler.get_effective_spacing()
        
        # Store spine coordinates in original space for future reference
        if path_id in self.state['spine_data']:
            original_spine_coords = self.scaler.unscale_coordinates(spine_positions)
            self.state['spine_data'][path_id]['coordinates_original_space'] = original_spine_coords
            self.state['spine_data'][path_id]['detection_spacing'] = spacing
        
        # Build comprehensive status message
        status_parts = [f"Detected {len(spine_positions)} spines for {path_data['name']}"]
        
        # Add processing method info
        if 'spine_data' in self.state and path_id in self.state['spine_data']:
            spine_info = self.state['spine_data'][path_id]
            if spine_info.get('detection_method') == 'memory_optimized_angle_based_extended':
                if spine_info.get('parameters', {}).get('enable_parallel', False):
                    status_parts.append("(optimized, parallel)")
                else:
                    status_parts.append("(optimized)")
        
        # Add path algorithm info
        if path_data.get('algorithm') == 'waypoint_astar':
            status_parts.append("(waypoint_astar path)")
        elif path_data.get('smoothed', False):
            status_parts.append("(smoothed path)")
        
        # Add spacing info
        status_parts.append(f"at {spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f} nm")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Store spine positions in state
        self.state['spine_positions'] = spine_positions
        
        # Enable spine segmentation for this path
        self.spine_segmentation_widget.enable_for_path(path_id)
    
    def on_spine_segmentation_completed(self, path_id, layer_name):
        """Handle when spine segmentation is completed for a path"""
        path_data = self.state['paths'][path_id]
        spacing = self.scaler.get_effective_spacing()
        
        # Build comprehensive completion message
        status_parts = [f"Spine segmentation completed for {path_data['name']}"]
        
        # Add algorithm details
        algorithm_details = []
        if path_data.get('algorithm') == 'waypoint_astar':
            algorithm_details.append("path tracing")
        
        if 'spine_data' in self.state and path_id in self.state['spine_data']:
            spine_info = self.state['spine_data'][path_id]
            if spine_info.get('detection_method') == 'memory_optimized_angle_based_extended':
                algorithm_details.append("spine detection")
        
        if algorithm_details:
            status_parts.append(f"({', '.join(algorithm_details)})")
        
        # Add spacing info
        status_parts.append(f"at {spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f} nm")
        
        self.path_info.setText(" ".join(status_parts))
        
        # Create comprehensive completion notification
        notification_parts = [f"Complete workflow finished for {path_data['name']}"]
        if algorithm_details:
            notification_parts.append(f"using {', '.join(algorithm_details)}")
        notification_parts.append(f"at current scaling")
        
        napari.utils.notifications.show_info(" ".join(notification_parts))
    
    def get_current_image(self):
        """Get the currently scaled image"""
        return self.current_image
        
    def get_current_spacing(self):
        """Get current voxel spacing in (x, y, z) format"""
        return self.scaler.get_effective_spacing()
        
    def scale_coordinates_to_original(self, coordinates):
        """
        Convert coordinates from current scaled space to original image space
        Useful for saving results that reference the original image
        """
        return self.scaler.unscale_coordinates(coordinates)
        
    def scale_coordinates_from_original(self, coordinates):
        """
        Convert coordinates from original image space to current scaled space
        Useful for loading previous results
        """
        return self.scaler.scale_coordinates(coordinates)