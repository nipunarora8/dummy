import napari
import numpy as np
import imageio as io
from napari.layers import Points, Image
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFileDialog, QTabWidget,
    QListWidget, QListWidgetItem, QFrame,
    QProgressBar, QCheckBox, QSpinBox
)
import sys
sys.path.append('../path_tracing/brightest-path-lib')
from brightest_path_lib.algorithm import BidirectionalAStarSearch, WaypointBidirectionalAStarSearch
import sys
import uuid
from segmentation_module import DendriteSegmenter


class BrightestPathWidget(QWidget):
    def __init__(self, viewer, image):
        super().__init__()
        self.viewer = viewer
        self.image = image
        
        # List to store multiple waypoints
        self.waypoints = []  
        
        # Track multiple paths
        self.paths = {}  # Dictionary to store multiple paths
        self.path_layers = {}  # Dictionary to store individual path layers
        self.next_path_number = 1  # For naming paths (Path 1, Path 2, etc.)
        self.color_idx = 0  # Index to cycle through colors
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Initialize the segmentation model (don't load it yet)
        self.segmenter = None
        self.segmentation_layer = None
        
        # Create layers
        self.image_layer = self.viewer.add_image(
            self.image, name='Image', colormap='gray'
        )
        
        # Create a waypoints layer instead of separate start/end layers
        self.waypoints_layer = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='Waypoints',
            size=15,
            face_color='cyan',
            symbol='x'
        )
        
        # Add a special layer for 3D traced path visualization
        if self.image.ndim > 2:
            self.traced_path_layer = self.viewer.add_points(
                np.empty((0, self.image.ndim)),
                name='Traced Path (3D)',
                size=4,
                face_color='magenta',
                opacity=0.7,
                visible=False  # Hidden by default until 3D path is calculated
            )
        
        # Set up UI
        self.setup_ui()
        
        # Set up event handling for points layers
        self.waypoints_layer.events.data.connect(self.on_waypoints_changed)
        
        # Default mode for waypoints layer
        self.waypoints_layer.mode = 'add'
        
        # By default, activate the waypoints layer to begin workflow
        self.viewer.layers.selection.active = self.waypoints_layer
        show_info("Waypoints layer activated. Click on the image to set waypoints.")

    def setup_ui(self):
        """Create the UI panel with controls"""
        try:
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
            
            # First tab: Point Selection
            point_selection_tab = QWidget()
            point_layout = QVBoxLayout()
            point_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            point_layout.setContentsMargins(2, 2, 2, 2)
            point_selection_tab.setLayout(point_layout)
            
            
            # Waypoints section
            waypoints_section = QWidget()
            waypoints_layout = QVBoxLayout()
            waypoints_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            waypoints_layout.setContentsMargins(2, 2, 2, 2)
            waypoints_section.setLayout(waypoints_layout)
            
            waypoints_instr = QLabel("1. Click on the image for points\n2. Now click on Find Path to find the prompt points\n3. Head to the Segmentation tab to run segmentation")
            waypoints_layout.addWidget(waypoints_instr)
            
            self.select_waypoints_btn = QPushButton("Select Waypoints Layer")
            self.select_waypoints_btn.setFixedHeight(22)
            self.select_waypoints_btn.clicked.connect(self.activate_waypoints_layer)
            waypoints_layout.addWidget(self.select_waypoints_btn)
            
            self.waypoints_status = QLabel("Status: No waypoints set")
            waypoints_layout.addWidget(self.waypoints_status)
            point_layout.addWidget(waypoints_section)
            
            # Find path button
            find_btns_layout = QHBoxLayout()
            find_btns_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            find_btns_layout.setContentsMargins(2, 2, 2, 2)
            
            self.find_path_btn = QPushButton("Find Path")
            self.find_path_btn.setFixedHeight(22)
            self.find_path_btn.clicked.connect(self.find_path)
            self.find_path_btn.setEnabled(False)
            find_btns_layout.addWidget(self.find_path_btn)
            
            point_layout.addLayout(find_btns_layout)
            
            # Trace Another Path button
            self.trace_another_btn = QPushButton("Trace Another Path")
            self.trace_another_btn.setFixedHeight(22)
            self.trace_another_btn.clicked.connect(self.trace_another_path)
            self.trace_another_btn.setEnabled(False)  # Disabled until we have a path
            point_layout.addWidget(self.trace_another_btn)
            
            # Clear points button
            self.clear_points_btn = QPushButton("Clear Points (Start Over)")
            self.clear_points_btn.setFixedHeight(22)
            self.clear_points_btn.clicked.connect(self.clear_points)
            point_layout.addWidget(self.clear_points_btn)
            
            # Status messages
            self.error_status = QLabel("")
            self.error_status.setStyleSheet("color: red;")
            point_layout.addWidget(self.error_status)
            
            # Second tab: Path Management
            path_management_tab = QWidget()
            path_layout = QVBoxLayout()
            path_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            path_layout.setContentsMargins(2, 2, 2, 2)
            path_management_tab.setLayout(path_layout)
            
            # Path list with instructions
            path_layout.addWidget(QLabel("Saved Paths (select two paths to connect them):"))
            self.path_list = QListWidget()
            self.path_list.setFixedHeight(80)
            self.path_list.setSelectionMode(QListWidget.ExtendedSelection)  # Allow multiple selection
            self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
            path_layout.addWidget(self.path_list)
            
            # Path management buttons
            path_buttons_layout = QHBoxLayout()
            path_buttons_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            path_buttons_layout.setContentsMargins(2, 2, 2, 2)
            
            self.view_path_btn = QPushButton("View Selected Path")
            self.view_path_btn.setFixedHeight(22)
            self.view_path_btn.clicked.connect(self.view_selected_path)
            self.view_path_btn.setEnabled(False)
            path_buttons_layout.addWidget(self.view_path_btn)
            
            self.delete_path_btn = QPushButton("Delete Selected Path(s)")
            self.delete_path_btn.setFixedHeight(22)
            self.delete_path_btn.clicked.connect(self.delete_selected_paths)
            self.delete_path_btn.setEnabled(False)
            path_buttons_layout.addWidget(self.delete_path_btn)
            
            path_layout.addLayout(path_buttons_layout)
            
            # Path connection button
            self.connect_paths_btn = QPushButton("Connect Selected Paths")
            self.connect_paths_btn.setFixedHeight(22)
            self.connect_paths_btn.setToolTip("Select exactly 2 paths to connect them")
            self.connect_paths_btn.clicked.connect(self.connect_selected_paths)
            self.connect_paths_btn.setEnabled(False)
            path_layout.addWidget(self.connect_paths_btn)
            
            # Path visibility options
            visibility_layout = QHBoxLayout()
            visibility_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            visibility_layout.setContentsMargins(2, 2, 2, 2)
            
            self.show_all_btn = QPushButton("Show All Paths")
            self.show_all_btn.setFixedHeight(22)
            self.show_all_btn.clicked.connect(lambda: self.set_paths_visibility(True))
            visibility_layout.addWidget(self.show_all_btn)
            
            self.hide_all_btn = QPushButton("Hide All Paths")
            self.hide_all_btn.setFixedHeight(22)
            self.hide_all_btn.clicked.connect(lambda: self.set_paths_visibility(False))
            visibility_layout.addWidget(self.hide_all_btn)
            
            path_layout.addLayout(visibility_layout)
            
            # Export button
            self.export_all_btn = QPushButton("Export All Paths")
            self.export_all_btn.setFixedHeight(22)
            self.export_all_btn.clicked.connect(self.export_all_paths)
            self.export_all_btn.setEnabled(False)
            path_layout.addWidget(self.export_all_btn)
            
            # Third tab: Segmentation
            segmentation_tab = QWidget()
            seg_layout = QVBoxLayout()
            seg_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            seg_layout.setContentsMargins(2, 2, 2, 2)
            segmentation_tab.setLayout(seg_layout)
            
            # Model settings
            seg_layout.addWidget(QLabel("<b>Dendrite Segmentation</b>"))
            seg_layout.addWidget(QLabel("1. Load Segmentation Model\n2. Choose the path you want to segment\n3. Click on Run Segmentation to Segment"))
            
            # Model paths
            model_section = QWidget()
            model_layout = QVBoxLayout()
            model_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            model_layout.setContentsMargins(2, 2, 2, 2)
            model_section.setLayout(model_layout)
            
            model_layout.addWidget(QLabel("Model Paths:"))
            self.model_path_edit = QLabel("SAM2 Model: checkpoints/sam2.1_hiera_small.pt")
            model_layout.addWidget(self.model_path_edit)
            
            self.config_path_edit = QLabel("Config: sam2.1_hiera_s.yaml")
            model_layout.addWidget(self.config_path_edit)
            
            self.weights_path_edit = QLabel("Weights: results/samv2_small_2025-03-06-17-13-15/model_22500.torch")
            model_layout.addWidget(self.weights_path_edit)
            
            seg_layout.addWidget(model_section)

            # Add separator
            separator2 = QFrame()
            separator2.setFrameShape(QFrame.HLine)
            separator2.setFrameShadow(QFrame.Sunken)
            seg_layout.addWidget(separator2)
            
            # Path selection for segmentation
            seg_layout.addWidget(QLabel("Select a path to segment:"))
            self.seg_path_list = QListWidget()
            self.seg_path_list.setFixedHeight(80)
            self.seg_path_list.itemSelectionChanged.connect(self.on_seg_path_selection_changed)
            seg_layout.addWidget(self.seg_path_list)
            
            # Segmentation parameters
            params_section = QWidget()
            params_layout = QVBoxLayout()
            params_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            params_layout.setContentsMargins(2, 2, 2, 2)
            params_section.setLayout(params_layout)
            
            params_layout.addWidget(QLabel("Segmentation Parameters:"))
            
            # Patch size
            patch_size_layout = QHBoxLayout()
            patch_size_layout.setSpacing(2)  # Replace some_layout with the actual variable name
            patch_size_layout.setContentsMargins(2, 2, 2, 2)
            patch_size_layout.addWidget(QLabel("Patch Size:"))
            self.patch_size_spin = QSpinBox()
            self.patch_size_spin.setRange(32, 512)
            self.patch_size_spin.setSingleStep(32)
            self.patch_size_spin.setValue(128)
            patch_size_layout.addWidget(self.patch_size_spin)
            params_layout.addLayout(patch_size_layout)
            
            # Frame range
            self.use_full_volume_cb = QCheckBox("Process Full Volume")
            self.use_full_volume_cb.setChecked(False)
            params_layout.addWidget(self.use_full_volume_cb)
            
            seg_layout.addWidget(params_section)
            
            # Add separator
            separator3 = QFrame()
            separator3.setFrameShape(QFrame.HLine)
            separator3.setFrameShadow(QFrame.Sunken)
            seg_layout.addWidget(separator3)
            
            # Load model and run segmentation buttons
            self.load_model_btn = QPushButton("Load Segmentation Model")
            self.load_model_btn.setFixedHeight(22)
            self.load_model_btn.clicked.connect(self.load_segmentation_model)
            seg_layout.addWidget(self.load_model_btn)
            
            self.run_segmentation_btn = QPushButton("Run Segmentation")
            self.run_segmentation_btn.setFixedHeight(22)
            self.run_segmentation_btn.clicked.connect(self.run_segmentation)
            self.run_segmentation_btn.setEnabled(False)  # Disabled until model is loaded
            seg_layout.addWidget(self.run_segmentation_btn)
            
            # Progress bar
            self.segmentation_progress = QProgressBar()
            self.segmentation_progress.setValue(0)
            seg_layout.addWidget(self.segmentation_progress)
            
            # Status message
            self.segmentation_status = QLabel("Status: Model not loaded")
            self.segmentation_status.setWordWrap(True)
            seg_layout.addWidget(self.segmentation_status)
            
            # Add tabs to widget
            self.tabs.addTab(point_selection_tab, "Point Selection")
            self.tabs.addTab(path_management_tab, "Path Management")
            self.tabs.addTab(segmentation_tab, "Segmentation")
            self.tabs.setTabBarAutoHide(True)
            self.tabs.setStyleSheet("QTabBar::tab { height: 22px; }")
            layout.addWidget(self.tabs)
            
            # Current path info at the bottom
            self.path_info = QLabel("Path: Not calculated")
            layout.addWidget(self.path_info)
        except Exception as e:
            print(f"Error setting up UI: {str(e)}")

    def activate_waypoints_layer(self):
        """Activate the waypoints layer for selecting"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.waypoints_layer
            self.error_status.setText("")
            show_info("Waypoints layer activated. Click on the image to set waypoints.")
        except Exception as e:
            error_msg = f"Error activating waypoints layer: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def on_waypoints_changed(self, event=None):
        """Handle when waypoints are added or changed"""
        # Prevent recursive function calls
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.waypoints_layer.data) > 0:
                # Check all points are within image bounds
                valid_points = []
                for point in self.waypoints_layer.data:
                    valid = True
                    for i, coord in enumerate(point):
                        if coord < 0 or coord >= self.image.shape[i]:
                            valid = False
                            break
                            
                    if valid:
                        valid_points.append(point)
                    
                # Update the waypoints layer with only valid points
                if len(valid_points) != len(self.waypoints_layer.data):
                    self.waypoints_layer.data = np.array(valid_points)
                    show_info("Some points were outside image bounds and were removed.")
                
                # Store valid waypoints
                self.waypoints = [point.astype(int) for point in valid_points]
                
                # Update status
                self.waypoints_status.setText(f"Status: {len(self.waypoints)} waypoints set")
                
                # Enable find path button if we have at least 2 waypoints
                self.find_path_btn.setEnabled(len(self.waypoints) >= 2)
            else:
                self.waypoints = []
                self.waypoints_status.setText("Status: No waypoints set")
                self.find_path_btn.setEnabled(False)
        except Exception as e:
            show_info(f"Error setting waypoints: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def find_distance(self, start_point, end_point):
        """
        Calculate the distance between two points in 3D space.
        
        Parameters:
        start_point (np.ndarray): Starting point.
        end_point (np.ndarray): Ending point.
        
        Returns:
        float: Distance between the two points.
        """
        return np.sqrt(np.sum((start_point - end_point) ** 2))
    
    def find_farthest_points(self, points):
        """
        Find the two points with the maximum distance between them.
        
        Parameters:
        points (list): List of points.
        
        Returns:
        tuple: Start point, end point, and remaining waypoints.
        """
        points = [np.array(p) for p in points]
        max_distance = 0
        start_point = None
        end_point = None
        
        # Find the two points that are farthest apart
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = self.find_distance(points[i], points[j])
                if dist > max_distance:
                    max_distance = dist
                    start_point = points[i].copy()
                    end_point = points[j].copy()
        
        # Remove the start and end points from the list to get the waypoints
        if start_point is not None and end_point is not None:
            waypoints = [p for p in points if not np.array_equal(p, start_point) and not np.array_equal(p, end_point)]
            return start_point, end_point, waypoints
        
        return None, None, []
    
    def trace_another_path(self):
        """Reset for tracing a new path while preserving existing paths"""
        # Clear current waypoints to start fresh
        self.waypoints = []
        self.waypoints_layer.data = np.empty((0, self.image.ndim))
        
        # Reset UI for new path
        self.waypoints_status.setText("Status: No waypoints set")
        self.path_info.setText("Path: Not calculated")
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        # Activate the waypoints layer for the new path
        self.viewer.layers.selection.active = self.waypoints_layer
        show_info("Ready to trace a new path. Click on the image to set waypoints.")
    
    def clear_points(self):
        """Clear all waypoints and path without saving"""
        self.waypoints = []
        self.waypoints_layer.data = np.empty((0, self.image.ndim))
        
        # Clear traced path layer if it exists
        if hasattr(self, 'traced_path_layer'):
            self.traced_path_layer.data = np.empty((0, self.image.ndim))
            self.traced_path_layer.visible = False
            
        # Reset UI
        self.waypoints_status.setText("Status: No waypoints set")
        self.path_info.setText("Path: Not calculated")
        
        # Reset buttons
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        self.run_segmentation_btn.setEnabled(False)
        
        show_info("All points and path cleared. Ready to start over.")

    def find_path(self):
        """Find the brightest path through the waypoints"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.waypoints) < 2:
                show_info("Please set at least 2 waypoints")
                self.error_status.setText("Error: Please set at least 2 waypoints")
                return
            
            # Clear any previous error messages
            self.error_status.setText("")
            show_info("Finding brightest path through waypoints...")
            self.path_info.setText("Path: Calculating...")
            
            # Find the two farthest points to use as start and end
            waypoints_copy = [point.copy() for point in self.waypoints]
            start_point, end_point, intermediate_waypoints = self.find_farthest_points(waypoints_copy)
            
            if start_point is None or end_point is None:
                show_info("Failed to determine start and end points")
                self.error_status.setText("Error: Failed to determine start and end points")
                return

            search_algorithm = WaypointBidirectionalAStarSearch(
                self.image, 
                start_point=start_point, 
                goal_point=end_point,
                waypoints=intermediate_waypoints
            )
            show_info(f"Using waypoint search with {len(intermediate_waypoints)} intermediate points")
            
            # Find the path
            brightest_path = search_algorithm.search()
            
            # Process the found path
            if brightest_path is not None and len(brightest_path) > 0:
                # Generate path name
                path_name = f"Path {self.next_path_number}"
                self.next_path_number += 1
                
                # Get color for this path
                path_color = self.get_next_color()
                
                # Create a new layer for this path
                path_data = np.array(brightest_path)
                path_layer = self.viewer.add_points(
                    path_data,
                    name=path_name,
                    size=3,
                    face_color=path_color,
                    opacity=0.8
                )
                
                # For 3D visualization, create a traced path that shows the whole path in every frame
                if self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                    # Get the z-range (frame range) that we need to span
                    z_values = [point[0] for point in brightest_path]
                    min_z = int(min(z_values))
                    max_z = int(max(z_values))
                    
                    # Create a projection of the path onto every frame in the range
                    traced_points = []
                    
                    # For each frame in our range
                    for z in range(min_z, max_z + 1):
                        # Add all path points to this frame by changing their z-coordinate
                        for point in brightest_path:
                            # Create a new point with the current frame's z-coordinate
                            new_point = point.copy()
                            new_point[0] = z  # Set the z-coordinate to the current frame
                            traced_points.append(new_point)
                    
                    # Update the traced path layer with all these points
                    self.traced_path_layer.data = np.array(traced_points)
                    self.traced_path_layer.visible = True
                    
                    # Navigate to the frame where the path starts to provide better initial view
                    self.viewer.dims.set_point(0, min_z)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the path with all its data
                self.paths[path_id] = {
                    'name': path_name,
                    'data': path_data,
                    'start': start_point.copy(),
                    'end': end_point.copy(),
                    'waypoints': [wp.copy() for wp in intermediate_waypoints] if intermediate_waypoints else [],
                    'visible': True,
                    'layer': path_layer
                }
                
                # Store reference to the layer
                self.path_layers[path_id] = path_layer
                
                # Add to path list
                item = QListWidgetItem(path_name)
                item.setData(100, path_id)  # Store path ID as custom data
                self.path_list.addItem(item)
                
                # Update UI
                msg = f"Path found with {len(brightest_path)} points, saved as {path_name}"
                show_info(msg)
                self.path_info.setText(f"Path: {msg}")
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # Enable export all button
                self.export_all_btn.setEnabled(True)

                self.current_path_id = path_id
                
                # If we have a path and the model is loaded, enable segmentation
                if hasattr(self, 'segmenter') and self.segmenter is not None:
                    self.run_segmentation_btn.setEnabled(True)

                seg_item = QListWidgetItem(path_name)
                seg_item.setData(100, path_id)
                self.seg_path_list.addItem(seg_item)
                    
            else:
                # No path found
                msg = "No path found"
                show_info(msg)
                self.path_info.setText(f"Path: {msg}")
                self.error_status.setText("Error: No path found between selected points")
        except Exception as e:
            msg = f"Error finding path: {e}"
            show_info(msg)
            self.path_info.setText(f"Path: Error - {str(e)}")
            self.error_status.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False

    def get_next_color(self):
        """Get the next color from the predefined list"""
        # List of named colors that work well for paths
        colors = ['cyan', 'magenta', 'green', 'blue', 'orange', 
                  'purple', 'teal', 'coral', 'gold', 'lavender']
        
        # Get the next color and increment the counter
        color = colors[self.color_idx % len(colors)]
        self.color_idx += 1
        
        return color
    
    def on_path_selection_changed(self):
        """Handle when path selection changes in the list"""
        # Prevent processing during updates
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            num_selected = len(selected_items)
            
            # Enable/disable buttons based on selection
            self.delete_path_btn.setEnabled(num_selected > 0)
            self.view_path_btn.setEnabled(num_selected == 1)
            self.connect_paths_btn.setEnabled(num_selected == 2)
        except Exception as e:
            show_info(f"Error handling selection change: {str(e)}")
        finally:
            self.handling_event = False
    
    def view_selected_path(self):
        """View the selected path from the list"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) != 1:
                return
                
            item = selected_items[0]
            path_id = item.data(100)
            if path_id in self.paths:
                path_data = self.paths[path_id]
                
                # Update waypoints with start, end, and intermediate waypoints
                new_waypoints = []
                if 'start' in path_data and path_data['start'] is not None:
                    new_waypoints.append(path_data['start'])
                    
                if 'waypoints' in path_data and path_data['waypoints']:
                    new_waypoints.extend(path_data['waypoints'])
                    
                if 'end' in path_data and path_data['end'] is not None:
                    new_waypoints.append(path_data['end'])
                
                # Update the waypoints layer
                if new_waypoints:
                    self.waypoints_layer.data = np.array(new_waypoints)
                    self.waypoints = new_waypoints
                    self.waypoints_status.setText(f"Status: {len(new_waypoints)} waypoints loaded")
                
                self.path_info.setText(f"Path: {path_data['name']} loaded with {len(path_data['data'])} points")
                
                # Store current path ID
                self.current_path_id = path_id
                
                # Enable find path button
                self.find_path_btn.setEnabled(len(new_waypoints) >= 2)
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # If we have a path and the model is loaded, enable segmentation
                if hasattr(self, 'segmenter') and self.segmenter is not None:
                    self.run_segmentation_btn.setEnabled(True)
                    
                # Switch to the Point Selection tab
                self.tabs.setCurrentIndex(0)
                
                # Ensure the selected path's layer is visible
                path_data['layer'].visible = True
                
                # Clear any error messages
                self.error_status.setText("")
                
                show_info(f"Loaded {path_data['name']}")
        except Exception as e:
            error_msg = f"Error viewing path: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def delete_selected_paths(self):
        """Delete the currently selected paths"""
        selected_items = self.path_list.selectedItems()
        if not selected_items:
            show_info("No paths selected")
            return
        
        paths_deleted = False
        
        for item in selected_items:
            # Get the path ID
            path_id = item.data(100)
            
            if path_id in self.paths:
                paths_deleted = True
                path_name = self.paths[path_id]['name']
                
                # Remove the path layer from viewer
                if path_id in self.path_layers:
                    self.viewer.layers.remove(self.path_layers[path_id])
                    del self.path_layers[path_id]
                
                # Remove corresponding segmentation layer if it exists
                seg_layer_name = f"Segmentation - {path_name}"
                for layer in list(self.viewer.layers):  # Create a copy of the list to safely modify during iteration
                    if layer.name == seg_layer_name:
                        self.viewer.layers.remove(layer)
                        if hasattr(self, 'segmentation_layer') and self.segmentation_layer.name == seg_layer_name:
                            self.segmentation_layer = None
                        show_info(f"Removed segmentation layer for {path_name}")
                        break
                
                # Remove from dictionary
                del self.paths[path_id]
                
                # Remove from list widget
                row = self.path_list.row(item)
                self.path_list.takeItem(row)
                
                # Also remove from segmentation path list
                for i in range(self.seg_path_list.count()):
                    seg_item = self.seg_path_list.item(i)
                    if seg_item.data(100) == path_id:
                        self.seg_path_list.takeItem(i)
                        break
                
                show_info(f"Deleted {path_name}")
        
        # Update traced path visualization if any paths were deleted
        if paths_deleted and self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
            self._update_traced_path_visualization()
        
        # Disable buttons if no paths remain
        if self.path_list.count() == 0:
            self.delete_path_btn.setEnabled(False)
            self.view_path_btn.setEnabled(False)
            self.connect_paths_btn.setEnabled(False)
            self.export_all_btn.setEnabled(False)
            self.run_segmentation_btn.setEnabled(False)
    
    def _update_traced_path_visualization(self):
        """Update the traced path visualization to reflect current paths"""
        if not hasattr(self, 'traced_path_layer') or self.traced_path_layer is None:
            return
            
        if not self.paths:
            # If no paths remain, clear the traced path layer
            self.traced_path_layer.data = np.empty((0, self.image.ndim))
            self.traced_path_layer.visible = False
            return
            
        # Create a comprehensive visualization of all paths in the traced layer
        all_traced_points = []
        
        # First, determine the full z-range for all paths
        min_z = float('inf')
        max_z = float('-inf')
        
        for path_id, path_data in self.paths.items():
            if len(path_data['data']) > 0 and path_data['visible']:
                z_values = [point[0] for point in path_data['data']]
                path_min_z = int(min(z_values))
                path_max_z = int(max(z_values))
                
                min_z = min(min_z, path_min_z)
                max_z = max(max_z, path_max_z)
        
        # If we have valid z-range
        if min_z != float('inf') and max_z != float('-inf'):
            # For each frame in the full range
            for z in range(min_z, max_z + 1):
                # Add all paths to this frame
                for path_id, path_data in self.paths.items():
                    if path_data['visible']:
                        for point in path_data['data']:
                            # Create a new point with the current frame's z-coordinate
                            new_point = point.copy()
                            new_point[0] = z  # Set the z-coordinate to the current frame
                            all_traced_points.append(new_point)
            
            # Update the traced path layer
            if all_traced_points:
                self.traced_path_layer.data = np.array(all_traced_points)
                self.traced_path_layer.visible = True
            else:
                self.traced_path_layer.data = np.empty((0, self.image.ndim))
                self.traced_path_layer.visible = False
    
    def set_paths_visibility(self, visible):
        """Set visibility of all saved path layers and update traced path visualization"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            # Show/hide individual path layers
            for path_id, layer in self.path_layers.items():
                layer.visible = visible
                self.paths[path_id]['visible'] = visible
            
            # Update traced path visualization
            if self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                if visible:
                    self._update_traced_path_visualization()
                else:
                    # Hide traced path layer when hiding all paths
                    self.traced_path_layer.data = np.empty((0, self.image.ndim))
                    self.traced_path_layer.visible = False
            
            action = "shown" if visible else "hidden"
            show_info(f"All paths {action}")
        except Exception as e:
            error_msg = f"Error updating path visibility: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def connect_selected_paths(self):
        """Connect two selected paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            
            if len(selected_items) != 2:
                show_info("Please select exactly two paths to connect")
                return
                
            # Get the path IDs
            path_id1 = selected_items[0].data(100)
            path_id2 = selected_items[1].data(100)
            
            if path_id1 not in self.paths or path_id2 not in self.paths:
                show_info("Invalid path selection")
                return
                
            # Get the path data
            path1 = self.paths[path_id1]
            path2 = self.paths[path_id2]
            
            # Check if paths have start/end points
            if path1['start'] is None or path2['end'] is None:
                show_info("Both paths must have start and end points to connect them")
                return
                
            # Get start of path1 and end of path2
            start_point = path1['start']
            end_point = path2['end']
            
            show_info(f"Connecting {path1['name']} to {path2['name']}...")
            
            # Determine if we're doing 2D or 3D search
            is_same_frame = True
            if self.image.ndim > 2:
                is_same_frame = start_point[0] == end_point[0]
                
            # Prepare points format based on 2D or 3D
            if is_same_frame and self.image.ndim > 2:
                # 2D case: use [y, x] format (ignore z)
                search_start = start_point[1:3]  # [y, x]
                search_end = end_point[1:3]      # [y, x]
                search_image = self.image[int(start_point[0])]
                show_info(f"Using 2D path search on frame {int(start_point[0])}")
            else:
                # 3D case or already 2D image: use full coordinates
                search_start = start_point
                search_end = end_point
                search_image = self.image
                show_info("Using 3D path search across frames")
                
            # Search for connecting path
            search_algorithm = BidirectionalAStarSearch(
                search_image, 
                start_point=search_start, 
                goal_point=search_end
            )
            
            connecting_path = search_algorithm.search()
            
            # If path found, create combined path
            if connecting_path is not None and len(connecting_path) > 0:
                # Fix coordinates if needed (2D case)
                if is_same_frame and self.image.ndim > 2:
                    z_val = start_point[0]
                    fixed_connecting_path = []
                    for point in connecting_path:
                        if len(point) == 2:  # [y, x]
                            fixed_connecting_path.append([z_val, point[0], point[1]])
                        else:
                            fixed_connecting_path.append(point)
                    connecting_path = fixed_connecting_path
                
                # Convert to numpy arrays
                path1_data = path1['data']
                path2_data = path2['data']
                connecting_data = np.array(connecting_path)
                
                # Create combined path
                combined_path = np.vstack([path1_data, connecting_data, path2_data])
                
                # Create a name for the combined path
                combined_name = f"{path1['name']} + {path2['name']}"
                
                # Get a color
                combined_color = self.get_next_color()
                
                # Create a new layer
                combined_layer = self.viewer.add_points(
                    combined_path,
                    name=combined_name,
                    size=3,
                    face_color=combined_color,
                    opacity=0.7
                )
                
                # Create traced path visualization for combined path
                # This is especially important when connecting 2D+3D paths
                if self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                    # Get the z-range for the entire combined path
                    z_values = [point[0] for point in combined_path]
                    min_z = int(min(z_values))
                    max_z = int(max(z_values))
                    
                    # If z-range spans more than one slice, generate traced path
                    if max_z > min_z:
                        # Create a projection of the path onto every frame in the range
                        traced_points = []
                        
                        # For each frame in our range
                        for z in range(min_z, max_z + 1):
                            # Add all path points to this frame by changing their z-coordinate
                            for point in combined_path:
                                # Create a new point with the current frame's z-coordinate
                                new_point = point.copy()
                                new_point[0] = z  # Set the z-coordinate to the current frame
                                traced_points.append(new_point)
                        
                        # Update the traced path layer with all these points
                        self.traced_path_layer.data = np.array(traced_points)
                        self.traced_path_layer.visible = True
                        
                        # Navigate to the frame where the path starts for better visibility
                        self.viewer.dims.set_point(0, min_z)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Combine waypoints from both paths
                combined_waypoints = []
                if 'waypoints' in path1 and path1['waypoints']:
                    combined_waypoints.extend(path1['waypoints'])
                if 'waypoints' in path2 and path2['waypoints']:
                    combined_waypoints.extend(path2['waypoints'])
                
                # Store the combined path
                self.paths[path_id] = {
                    'name': combined_name,
                    'data': combined_path,
                    'start': path1['start'].copy(),
                    'end': path2['end'].copy(),
                    'waypoints': combined_waypoints,
                    'visible': True,
                    'layer': combined_layer
                }
                
                # Store reference to the layer
                self.path_layers[path_id] = combined_layer
                
                # Add to path list
                item = QListWidgetItem(combined_name)
                item.setData(100, path_id)
                self.path_list.addItem(item)
                
                # Add to segmentation path list
                seg_item = QListWidgetItem(combined_name)
                seg_item.setData(100, path_id)
                self.seg_path_list.addItem(seg_item)
                
                # Select the new path
                self.path_list.setCurrentItem(item)
                
                # Update UI
                msg = f"Connected {path1['name']} to {path2['name']} successfully"
                show_info(msg)
                
                # Update waypoints
                new_waypoints = []
                if path1['start'] is not None:
                    new_waypoints.append(path1['start'].copy())
                if 'waypoints' in path1 and path1['waypoints']:
                    new_waypoints.extend(path1['waypoints'])
                if 'waypoints' in path2 and path2['waypoints']:
                    new_waypoints.extend(path2['waypoints'])
                if path2['end'] is not None:
                    new_waypoints.append(path2['end'].copy())
                
                self.waypoints = new_waypoints
                self.waypoints_layer.data = np.array(new_waypoints)
                
                # Update waypoints status
                self.waypoints_status.setText(f"Status: {len(new_waypoints)} waypoints set")
                self.path_info.setText(f"Path: {combined_name} with {len(combined_path)} points")
                
                # Store current path ID
                self.current_path_id = path_id
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # If we have a path and the model is loaded, enable segmentation
                if hasattr(self, 'segmenter') and self.segmenter is not None:
                    self.run_segmentation_btn.setEnabled(True)
                
                # Clear any error messages
                self.error_status.setText("")
            else:
                error_msg = f"Could not find a path connecting {path1['name']} to {path2['name']}"
                show_info(error_msg)
                self.error_status.setText(error_msg)
        except Exception as e:
            error_msg = f"Error connecting paths: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
            
    def export_all_paths(self):
        """Export all paths to a file"""
        if not self.paths:
            show_info("No paths to export")
            return
        
        # Get path to save file
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save All Paths", "", "NumPy Files (*.npz)"
        )
        
        if not filepath:
            return
        
        try:
            # Prepare data for export
            path_data = {}
            for path_id, path_info in self.paths.items():
                export_data = {
                    'points': path_info['data'],
                    'start': path_info['start'] if 'start' in path_info and path_info['start'] is not None else np.array([]),
                    'end': path_info['end'] if 'end' in path_info and path_info['end'] is not None else np.array([]),
                }
                
                # Include waypoints if available
                if 'waypoints' in path_info and path_info['waypoints']:
                    export_data['waypoints'] = np.array(path_info['waypoints'])
                
                path_data[path_info['name']] = export_data
            
            # Save as NumPy archive
            np.savez(filepath, paths=path_data)
            
            show_info(f"All paths saved to {filepath}")
            
        except Exception as e:
            show_info(f"Error saving paths: {e}")
    
    def load_segmentation_model(self):
        """Load the segmentation model"""
        try:
            # Update status
            self.segmentation_status.setText("Status: Loading model...")
            self.load_model_btn.setEnabled(False)
            
            # Initialize segmenter if not already done
            if self.segmenter is None:
                self.segmenter = DendriteSegmenter(
                    model_path="checkpoints/sam2.1_hiera_small.pt",
                    config_path="sam2.1_hiera_s.yaml",
                    weights_path="results/samv2_small_2025-03-06-17-13-15/model_22500.torch"
                )
            
            # Load the model
            success = self.segmenter.load_model()
            
            if success:
                self.segmentation_status.setText("Status: Model loaded successfully!")
                self.run_segmentation_btn.setEnabled(len(self.paths) > 0)
                show_info("Segmentation model loaded successfully")
            else:
                self.segmentation_status.setText("Status: Failed to load model. Check console for errors.")
                self.load_model_btn.setEnabled(True)
                show_info("Failed to load segmentation model")
                
        except Exception as e:
            error_msg = f"Error loading segmentation model: {str(e)}"
            self.segmentation_status.setText(f"Status: {error_msg}")
            self.load_model_btn.setEnabled(True)
            show_info(error_msg)
            print(f"Error details: {str(e)}")

    def on_seg_path_selection_changed(self):
        """Handle when path selection changes in the segmentation tab"""
        # Prevent processing during updates
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.seg_path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.paths:
                    # Store the selected path ID for segmentation
                    self.selected_seg_path_id = path_id
                    path_name = self.paths[path_id]['name']
                    self.segmentation_status.setText(f"Status: Path '{path_name}' selected for segmentation")
                    
                    # Enable the segmentation button if the model is loaded
                    if hasattr(self, 'segmenter') and self.segmenter is not None:
                        self.run_segmentation_btn.setEnabled(True)
            else:
                # No path selected
                if hasattr(self, 'selected_seg_path_id'):
                    delattr(self, 'selected_seg_path_id')
                self.run_segmentation_btn.setEnabled(False)
        except Exception as e:
            show_info(f"Error handling segmentation path selection: {str(e)}")
        finally:
            self.handling_event = False

    def run_segmentation(self):
        """Run segmentation on the selected path"""
        if self.segmenter is None:
            show_info("Please load the segmentation model first")
            return
        
        if len(self.paths) == 0:
            show_info("Please create a path first")
            return
        
        try:
            # Get the selected path
            path_id = None
            if hasattr(self, 'selected_seg_path_id'):
                path_id = self.selected_seg_path_id
            else:
                # Fallback to the path selected in the list
                selected_items = self.seg_path_list.selectedItems()
                if len(selected_items) == 1:
                    path_id = selected_items[0].data(100)
            
            if path_id is None or path_id not in self.paths:
                show_info("Please select a path for segmentation")
                self.segmentation_status.setText("Status: No path selected for segmentation")
                return
            
            # Get the path data
            path_data = self.paths[path_id]['data']
            path_name = self.paths[path_id]['name']
            
            # Update UI
            self.segmentation_status.setText(f"Status: Running segmentation on {path_name}...")
            self.segmentation_progress.setValue(0)
            self.run_segmentation_btn.setEnabled(False)
            
            # Get segmentation parameters
            patch_size = self.patch_size_spin.value()
            use_full_volume = self.use_full_volume_cb.isChecked()
            
            # Determine volume range
            if use_full_volume:
                start_frame = 0
                end_frame = len(self.image) - 1
            else:
                # Use the range from the path
                z_values = [point[0] for point in path_data]
                start_frame = int(min(z_values))
                end_frame = int(max(z_values))
            
            print(f"Segmenting path '{path_name}' from frame {start_frame} to {end_frame}")
            print(f"Path has {len(path_data)} points")
            
            # Progress callback function
            def update_progress(current, total):
                progress = int((current / total) * 100)
                self.segmentation_progress.setValue(progress)
            
            # Try to run the segmentation
            result_masks = None
            result_masks = self.segmenter.process_volume(
                image=self.image,
                brightest_path=path_data,
                start_frame=start_frame,
                end_frame=end_frame,
                patch_size=patch_size,
                progress_callback=update_progress
            )
            
            # Process the results
            if result_masks is not None:
                # Ensure masks are binary (0 or 1)
                binary_masks = (result_masks > 0).astype(np.uint8)
                
                # Create or update the segmentation layer
                seg_layer_name = f"Segmentation - {path_name}"
                
                # Remove existing layer if it exists
                existing_layer = None
                for layer in self.viewer.layers:
                    if layer.name == seg_layer_name:
                        existing_layer = layer
                        break
                
                if existing_layer is not None:
                    print(f"Removing existing segmentation layer: {seg_layer_name}")
                    self.viewer.layers.remove(existing_layer)
                
                # Add the new segmentation layer with distinct appearance
                print(f"Adding new segmentation layer: {seg_layer_name}")
                print(f"Result masks shape: {binary_masks.shape}")
                print(f"Result masks type: {binary_masks.dtype}")
                print(f"Result masks min/max: {binary_masks.min()}/{binary_masks.max()}")
                
                # Add segmentation as labels instead of image for better visualization
                self.segmentation_layer = self.viewer.add_labels(
                    binary_masks,
                    name=seg_layer_name,
                    opacity=0.6
                )
                
                # Make sure the layer is visible
                if hasattr(self, 'segmentation_layer') and self.segmentation_layer in self.viewer.layers:
                    self.segmentation_layer.visible = True
                    print(f"Layer {seg_layer_name} is visible")
                
                # Update the viewer to refresh display
                self.viewer.reset_view()
                
                self.segmentation_status.setText(f"Status: Segmentation complete for {path_name}")
                show_info(f"Segmentation complete for {path_name}")
            else:
                self.segmentation_status.setText("Status: Segmentation failed. Check console for errors.")
                show_info("Segmentation failed")
        
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.segmentation_status.setText(f"Status: {error_msg}")
            show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.segmentation_progress.setValue(100)
            self.run_segmentation_btn.setEnabled(True)