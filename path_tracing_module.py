import napari
import numpy as np
import uuid
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame
)
from qtpy.QtCore import Signal
import sys
sys.path.append('../path_tracing/brightest-path-lib')
from brightest_path_lib.algorithm import BidirectionalAStarSearch, WaypointBidirectionalAStarSearch


class PathTracingWidget(QWidget):
    """Widget for tracing the brightest path through waypoints in an image."""
    
    # Define signals
    path_created = Signal(str, str, object)  # path_id, path_name, path_data
    path_updated = Signal(str, str, object)  # path_id, path_name, path_data
    
    def __init__(self, viewer, image, state):
        """Initialize the path tracing widget.
        
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
        
        # List to store waypoints
        self.waypoints = []
        
        # Settings for path finding
        self.next_path_number = 1
        self.color_idx = 0
        
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
        
        # Waypoints section
        waypoints_section = QWidget()
        waypoints_layout = QVBoxLayout()
        waypoints_layout.setSpacing(2)
        waypoints_layout.setContentsMargins(2, 2, 2, 2)
        waypoints_section.setLayout(waypoints_layout)
        
        waypoints_instr = QLabel("1. Click on the image for points\n2. Now click on Find Path to find the brightest path\n3. Head to other tabs for additional analyses")
        waypoints_layout.addWidget(waypoints_instr)
        
        self.select_waypoints_btn = QPushButton("Select Waypoints Layer")
        self.select_waypoints_btn.setFixedHeight(22)
        self.select_waypoints_btn.clicked.connect(self.activate_waypoints_layer)
        waypoints_layout.addWidget(self.select_waypoints_btn)
        
        self.waypoints_status = QLabel("Status: No waypoints set")
        waypoints_layout.addWidget(self.waypoints_status)
        layout.addWidget(waypoints_section)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Find path button
        find_btns_layout = QHBoxLayout()
        find_btns_layout.setSpacing(2)
        find_btns_layout.setContentsMargins(2, 2, 2, 2)
        
        self.find_path_btn = QPushButton("Find Path")
        self.find_path_btn.setFixedHeight(22)
        self.find_path_btn.clicked.connect(self.find_path)
        self.find_path_btn.setEnabled(False)
        find_btns_layout.addWidget(self.find_path_btn)
        
        layout.addLayout(find_btns_layout)
        
        # Trace Another Path button
        self.trace_another_btn = QPushButton("Trace Another Path")
        self.trace_another_btn.setFixedHeight(22)
        self.trace_another_btn.clicked.connect(self.trace_another_path)
        self.trace_another_btn.setEnabled(False)
        layout.addWidget(self.trace_another_btn)
        
        # Clear points button
        self.clear_points_btn = QPushButton("Clear Points (Start Over)")
        self.clear_points_btn.setFixedHeight(22)
        self.clear_points_btn.clicked.connect(self.clear_points)
        layout.addWidget(self.clear_points_btn)
        
        # Status messages
        self.error_status = QLabel("")
        self.error_status.setStyleSheet("color: red;")
        layout.addWidget(self.error_status)
    
    def activate_waypoints_layer(self):
        """Activate the waypoints layer for selecting"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.state['waypoints_layer']
            self.error_status.setText("")
            napari.utils.notifications.show_info("Waypoints layer activated. Click on the image to set waypoints.")
        except Exception as e:
            error_msg = f"Error activating waypoints layer: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
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
            
            waypoints_layer = self.state['waypoints_layer']
            if len(waypoints_layer.data) > 0:
                # Check all points are within image bounds
                valid_points = []
                for point in waypoints_layer.data:
                    valid = True
                    for i, coord in enumerate(point):
                        if coord < 0 or coord >= self.image.shape[i]:
                            valid = False
                            break
                            
                    if valid:
                        valid_points.append(point)
                    
                # Update the waypoints layer with only valid points
                if len(valid_points) != len(waypoints_layer.data):
                    waypoints_layer.data = np.array(valid_points)
                    napari.utils.notifications.show_info("Some points were outside image bounds and were removed.")
                
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
            napari.utils.notifications.show_info(f"Error setting waypoints: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def find_distance(self, start_point, end_point):
        """Calculate the distance between two points in 3D space."""
        return np.sqrt(np.sum((start_point - end_point) ** 2))
    
    def find_farthest_points(self, points):
        """Find the two points with the maximum distance between them."""
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
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Reset UI for new path
        self.waypoints_status.setText("Status: No waypoints set")
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        # Activate the waypoints layer for the new path
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Ready to trace a new path. Click on the image to set waypoints.")
    
    def clear_points(self):
        """Clear all waypoints and path without saving"""
        self.waypoints = []
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Clear traced path layer if it exists
        if self.state['traced_path_layer'] is not None:
            self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
            self.state['traced_path_layer'].visible = False
            
        # Reset UI
        self.waypoints_status.setText("Status: No waypoints set")
        
        # Reset buttons
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        napari.utils.notifications.show_info("All points and path cleared. Ready to start over.")
    
    def find_path(self):
        """Find the brightest path through the waypoints"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.waypoints) < 2:
                napari.utils.notifications.show_info("Please set at least 2 waypoints")
                self.error_status.setText("Error: Please set at least 2 waypoints")
                return
            
            # Clear any previous error messages
            self.error_status.setText("")
            napari.utils.notifications.show_info("Finding brightest path through waypoints...")
            
            # Find the two farthest points to use as start and end
            waypoints_copy = [point.copy() for point in self.waypoints]
            start_point, end_point, intermediate_waypoints = self.find_farthest_points(waypoints_copy)
            
            if start_point is None or end_point is None:
                napari.utils.notifications.show_info("Failed to determine start and end points")
                self.error_status.setText("Error: Failed to determine start and end points")
                return

            search_algorithm = WaypointBidirectionalAStarSearch(
                self.image, 
                start_point=start_point, 
                goal_point=end_point,
                waypoints=intermediate_waypoints
            )
            napari.utils.notifications.show_info(f"Using waypoint search with {len(intermediate_waypoints)} intermediate points")
            
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
                if self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
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
                    self.state['traced_path_layer'].data = np.array(traced_points)
                    self.state['traced_path_layer'].visible = True
                    
                    # Navigate to the frame where the path starts to provide better initial view
                    self.viewer.dims.set_point(0, min_z)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the path with all its data
                self.state['paths'][path_id] = {
                    'name': path_name,
                    'data': path_data,
                    'start': start_point.copy(),
                    'end': end_point.copy(),
                    'waypoints': [wp.copy() for wp in intermediate_waypoints] if intermediate_waypoints else [],
                    'visible': True,
                    'layer': path_layer
                }
                
                # Store reference to the layer
                self.state['path_layers'][path_id] = path_layer
                
                # Update UI
                msg = f"Path found with {len(brightest_path)} points, saved as {path_name}"
                napari.utils.notifications.show_info(msg)
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # Store current path ID in state
                self.state['current_path_id'] = path_id
                
                # Emit signal that a new path was created
                self.path_created.emit(path_id, path_name, path_data)
                    
            else:
                # No path found
                msg = "No path found"
                napari.utils.notifications.show_info(msg)
                self.error_status.setText("Error: No path found between selected points")
        except Exception as e:
            msg = f"Error finding path: {e}"
            napari.utils.notifications.show_info(msg)
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
    
    def load_path_waypoints(self, path_id):
        """Load the waypoints for a specific path"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if path_id not in self.state['paths']:
                return
                
            path_data = self.state['paths'][path_id]
            
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
                self.state['waypoints_layer'].data = np.array(new_waypoints)
                self.waypoints = new_waypoints
                self.waypoints_status.setText(f"Status: {len(new_waypoints)} waypoints loaded")
            
            # Enable find path button
            self.find_path_btn.setEnabled(len(new_waypoints) >= 2)
            
            # Enable trace another path button
            self.trace_another_btn.setEnabled(True)
                
            # Clear any error messages
            self.error_status.setText("")
            
            napari.utils.notifications.show_info(f"Loaded {path_data['name']}")
        except Exception as e:
            error_msg = f"Error loading path waypoints: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False