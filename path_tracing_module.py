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
from brightest_path_lib.algorithm import EnhancedWaypointAStarSearch


class PathTracingWidget(QWidget):
    """Widget for tracing the brightest path with enhanced algorithm backend."""
    
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
        
        # List to store waypoints as they are clicked
        self.clicked_points = []
        
        # Settings for path finding
        self.next_path_number = 1
        self.color_idx = 0
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI (keeping original interface)
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI panel with controls (original interface)"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Main instruction
        title = QLabel("<b>Path Tracing</b>")
        layout.addWidget(title)
        
        # Instructions section
        instructions_section = QWidget()
        instructions_layout = QVBoxLayout()
        instructions_layout.setSpacing(2)
        instructions_layout.setContentsMargins(2, 2, 2, 2)
        instructions_section.setLayout(instructions_layout)
        
        instructions = QLabel(
            "Instructions:\n"
            "1. Click points on the dendrite structure\n"
            "2. Click 'Find Path' to trace the brightest path\n"
            "3. Use 'Trace Another Path' for additional paths"
        )
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        layout.addWidget(instructions_section)
        
        # Waypoint controls section
        waypoints_section = QWidget()
        waypoints_layout = QVBoxLayout()
        waypoints_layout.setSpacing(2)
        waypoints_layout.setContentsMargins(2, 2, 2, 2)
        waypoints_section.setLayout(waypoints_layout)
        
        self.select_waypoints_btn = QPushButton("Start Point Selection")
        self.select_waypoints_btn.setFixedHeight(22)
        self.select_waypoints_btn.clicked.connect(self.activate_waypoints_layer)
        waypoints_layout.addWidget(self.select_waypoints_btn)
        
        self.waypoints_status = QLabel("Status: Click to start selecting points")
        waypoints_layout.addWidget(self.waypoints_status)
        layout.addWidget(waypoints_section)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(2)
        
        # Main path finding button (same as before)
        self.find_path_btn = QPushButton("Find Path")
        self.find_path_btn.setFixedHeight(26)
        self.find_path_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.find_path_btn.clicked.connect(self.find_path)
        self.find_path_btn.setEnabled(False)
        buttons_layout.addWidget(self.find_path_btn)
        
        layout.addLayout(buttons_layout)
        
        # Management buttons
        management_layout = QHBoxLayout()
        management_layout.setSpacing(2)
        
        self.trace_another_btn = QPushButton("Trace Another Path")
        self.trace_another_btn.setFixedHeight(22)
        self.trace_another_btn.clicked.connect(self.trace_another_path)
        self.trace_another_btn.setEnabled(False)
        management_layout.addWidget(self.trace_another_btn)
        
        self.clear_points_btn = QPushButton("Clear All Points")
        self.clear_points_btn.setFixedHeight(22)
        self.clear_points_btn.clicked.connect(self.clear_points)
        management_layout.addWidget(self.clear_points_btn)
        
        layout.addLayout(management_layout)
        
        # Status messages
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.error_status = QLabel("")
        self.error_status.setStyleSheet("color: red;")
        layout.addWidget(self.error_status)
    
    def activate_waypoints_layer(self):
        """Activate the waypoints layer for selecting points"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.state['waypoints_layer']
            self.error_status.setText("")
            self.status_label.setText("Click points on the dendrite structure")
            napari.utils.notifications.show_info("Click points on the dendrite")
        except Exception as e:
            error_msg = f"Error activating waypoints layer: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def on_waypoints_changed(self, event=None):
        """Handle when waypoints are added or changed"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            waypoints_layer = self.state['waypoints_layer']
            if len(waypoints_layer.data) > 0:
                # Validate points are within image bounds
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
                
                # Convert to integer coordinates and store
                self.clicked_points = [point.astype(int) for point in valid_points]
                
                # Update status
                num_points = len(self.clicked_points)
                self.waypoints_status.setText(f"Status: {num_points} points selected")
                
                # Enable buttons if we have enough points
                self.find_path_btn.setEnabled(num_points >= 2)
                
                if num_points >= 2:
                    self.status_label.setText("Ready to find path!")
                else:
                    self.status_label.setText(f"Need at least 2 points (currently have {num_points})")
            else:
                self.clicked_points = []
                self.waypoints_status.setText("Status: Click to start selecting points")
                self.find_path_btn.setEnabled(False)
                self.status_label.setText("")
        except Exception as e:
            napari.utils.notifications.show_info(f"Error processing waypoints: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def find_path(self):
        """Find path using the enhanced algorithm (backend only change)"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.clicked_points) < 2:
                napari.utils.notifications.show_info("Please select at least 2 points")
                self.error_status.setText("Error: Please select at least 2 points")
                return
            
            # Clear any previous error messages
            self.error_status.setText("")
            self.status_label.setText("Finding path...")
            
            # Convert clicked points to the format expected by the algorithm
            points_list = [point.tolist() for point in self.clicked_points]
            
            napari.utils.notifications.show_info("Finding brightest path...")
            
            # Use enhanced algorithm in backend (but user doesn't see the difference)
            search = EnhancedWaypointAStarSearch(
                image=self.image,
                points_list=points_list,
                intensity_threshold=0.3,  # Fixed sensible default
                auto_z_detection=True,    # Always enabled for better results
                waypoint_z_optimization=True,  # Always enabled for better results
                verbose=True  # Keep quiet unless debugging
            )
            
            # Run the search
            path, start_point, goal_point, processed_waypoints = search.search()
            
            # Check if path was found
            if search.found_path and len(path) > 0:
                # Generate path name
                path_name = f"Path {self.next_path_number}"
                self.next_path_number += 1
                
                # Get color for this path
                path_color = self.get_next_color()
                
                # Create a new layer for this path
                path_data = np.array(path)
                path_layer = self.viewer.add_points(
                    path_data,
                    name=path_name,
                    size=3,
                    face_color=path_color,
                    opacity=0.8
                )
                
                # Update 3D visualization if applicable
                if self.image.ndim > 2 and self.state['traced_path_layer'] is not None:
                    self._update_traced_path_visualization(path)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the path (enhanced backend data but simplified frontend)
                self.state['paths'][path_id] = {
                    'name': path_name,
                    'data': path_data,
                    'start': start_point.copy(),
                    'end': goal_point.copy(),
                    'waypoints': [wp.copy() for wp in processed_waypoints] if processed_waypoints else [],
                    'visible': True,
                    'layer': path_layer,
                    'original_clicks': [point.copy() for point in self.clicked_points]
                }
                
                # Store reference to the layer
                self.state['path_layers'][path_id] = path_layer
                
                # Update UI
                msg = f"Path found: {len(path)} points"
                napari.utils.notifications.show_info(msg)
                self.status_label.setText(f"Success: {path_name} created")
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # Store current path ID in state
                self.state['current_path_id'] = path_id
                
                # Emit signal that a new path was created
                self.path_created.emit(path_id, path_name, path_data)
                    
            else:
                # No path found
                msg = "Could not find a path"
                napari.utils.notifications.show_info(msg)
                self.error_status.setText("Error: No path found")
                self.status_label.setText("Try selecting different points")
                
        except Exception as e:
            msg = f"Error in path finding: {e}"
            napari.utils.notifications.show_info(msg)
            self.error_status.setText(f"Error: {str(e)}")
            print(f"Path finding error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.handling_event = False
    
    def _update_traced_path_visualization(self, path):
        """Update the 3D traced path visualization"""
        if self.state['traced_path_layer'] is None:
            return
            
        try:
            # Get the z-range of the path
            path_array = np.array(path)
            z_values = [point[0] for point in path]
            min_z = int(min(z_values))
            max_z = int(max(z_values))
            
            # Create a projection of the path onto every frame in the range
            traced_points = []
            for z in range(min_z, max_z + 1):
                for point in path:
                    new_point = point.copy()
                    new_point[0] = z
                    traced_points.append(new_point)
            
            # Update the traced path layer
            if traced_points:
                self.state['traced_path_layer'].data = np.array(traced_points)
                self.state['traced_path_layer'].visible = True
                self.viewer.dims.set_point(0, min_z)
        except Exception as e:
            print(f"Error updating traced path visualization: {e}")
    
    def trace_another_path(self):
        """Reset for tracing a new path while preserving existing paths"""
        # Clear current points
        self.clicked_points = []
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Reset UI for new path
        self.waypoints_status.setText("Status: Click to start selecting points")
        self.status_label.setText("Ready for new path - click points on dendrite")
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        # Activate the waypoints layer for the new path
        self.viewer.layers.selection.active = self.state['waypoints_layer']
        napari.utils.notifications.show_info("Ready to trace a new path. Click points on the dendrite.")
    
    def clear_points(self):
        """Clear all waypoints and paths"""
        self.clicked_points = []
        self.state['waypoints_layer'].data = np.empty((0, self.image.ndim))
        
        # Clear traced path layer if it exists
        if self.state['traced_path_layer'] is not None:
            self.state['traced_path_layer'].data = np.empty((0, self.image.ndim))
            self.state['traced_path_layer'].visible = False
            
        # Reset UI
        self.waypoints_status.setText("Status: Click to start selecting points")
        self.status_label.setText("")
        self.error_status.setText("")
        
        # Reset buttons
        self.find_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
        
        napari.utils.notifications.show_info("All points cleared. Ready to start over.")
    
    def get_next_color(self):
        """Get the next color from the predefined list"""
        colors = ['cyan', 'magenta', 'green', 'blue', 'orange', 
                  'purple', 'teal', 'coral', 'gold', 'lavender']
        
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
            
            # Check if this path has original clicks stored
            if 'original_clicks' in path_data:
                # Load the original clicked points
                self.clicked_points = [np.array(point) for point in path_data['original_clicks']]
                
                # Update the waypoints layer
                if self.clicked_points:
                    self.state['waypoints_layer'].data = np.array(self.clicked_points)
                    self.waypoints_status.setText(f"Status: {len(self.clicked_points)} points loaded")
            else:
                # Fallback - reconstruct from start, waypoints, and end
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
                    self.clicked_points = new_waypoints
                    self.waypoints_status.setText(f"Status: {len(new_waypoints)} points loaded")
            
            # Enable buttons
            if len(self.clicked_points) >= 2:
                self.find_path_btn.setEnabled(True)
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