import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QCheckBox, QSpinBox
)
from qtpy.QtCore import Signal
from segmentation_model import DendriteSegmenter


class SegmentationWidget(QWidget):
    """Widget for performing segmentation on brightest paths"""
    
    # Define signals
    segmentation_completed = Signal(str, str)  # path_id, layer_name
    
    def __init__(self, viewer, image, state):
        """Initialize the segmentation widget.
        
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
        
        # Initialize the segmentation model (don't load it yet)
        self.segmenter = None
        
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
        
        # Model settings
        layout.addWidget(QLabel("<b>Dendrite Segmentation</b>"))
        layout.addWidget(QLabel("1. Load Segmentation Model\n2. Choose the path you want to segment\n3. Click on Run Segmentation to Segment"))
        
        # Model paths
        model_section = QWidget()
        model_layout = QVBoxLayout()
        model_layout.setSpacing(2)
        model_layout.setContentsMargins(2, 2, 2, 2)
        model_section.setLayout(model_layout)
        
        model_layout.addWidget(QLabel("Model Paths:"))
        self.model_path_edit = QLabel("SAM2 Model: checkpoints/sam2.1_hiera_small.pt")
        model_layout.addWidget(self.model_path_edit)
        
        self.config_path_edit = QLabel("Config: sam2.1_hiera_s.yaml")
        model_layout.addWidget(self.config_path_edit)
        
        self.weights_path_edit = QLabel("Weights: results/samv2_small_2025-03-06-17-13-15/model_22500.torch")
        model_layout.addWidget(self.weights_path_edit)
        
        layout.addWidget(model_section)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Path selection for segmentation
        layout.addWidget(QLabel("Select a path to segment:"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(80)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Segmentation parameters
        params_section = QWidget()
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(2, 2, 2, 2)
        params_section.setLayout(params_layout)
        
        params_layout.addWidget(QLabel("Segmentation Parameters:"))
        
        # Patch size
        patch_size_layout = QHBoxLayout()
        patch_size_layout.setSpacing(2)
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
        
        layout.addWidget(params_section)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Load model and run segmentation buttons
        self.load_model_btn = QPushButton("Load Segmentation Model")
        self.load_model_btn.setFixedHeight(22)
        self.load_model_btn.clicked.connect(self.load_segmentation_model)
        layout.addWidget(self.load_model_btn)
        
        self.run_segmentation_btn = QPushButton("Run Segmentation")
        self.run_segmentation_btn.setFixedHeight(22)
        self.run_segmentation_btn.clicked.connect(self.run_segmentation)
        self.run_segmentation_btn.setEnabled(False)  # Disabled until model is loaded
        layout.addWidget(self.run_segmentation_btn)
        
        # Progress bar
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setValue(0)
        layout.addWidget(self.segmentation_progress)
        
        # Status message
        self.status_label = QLabel("Status: Model not loaded")
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
            
            # Add paths to list
            for path_id, path_data in self.state['paths'].items():
                item = QListWidgetItem(path_data['name'])
                item.setData(100, path_id)  # Store path ID as custom data
                self.path_list.addItem(item)
            
            # Enable segmentation button if model is loaded and a path is selected
            self.run_segmentation_btn.setEnabled(
                self.segmenter is not None and 
                self.path_list.count() > 0 and
                self.path_list.currentRow() >= 0
            )
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating path list: {str(e)}")
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
                    # Store the selected path ID for segmentation
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    self.status_label.setText(f"Status: Path '{path_name}' selected for segmentation")
                    
                    # Enable the segmentation button if the model is loaded
                    if self.segmenter is not None:
                        self.run_segmentation_btn.setEnabled(True)
            else:
                # No path selected
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.run_segmentation_btn.setEnabled(False)
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling segmentation path selection: {str(e)}")
        finally:
            self.handling_event = False
    
    def load_segmentation_model(self):
        """Load the segmentation model"""
        try:
            # Update status
            self.status_label.setText("Status: Loading model...")
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
                self.status_label.setText("Status: Model loaded successfully!")
                self.run_segmentation_btn.setEnabled(len(self.state['paths']) > 0 and hasattr(self, 'selected_path_id'))
                napari.utils.notifications.show_info("Segmentation model loaded successfully")
            else:
                self.status_label.setText("Status: Failed to load model. Check console for errors.")
                self.load_model_btn.setEnabled(True)
                napari.utils.notifications.show_info("Failed to load segmentation model")
                
        except Exception as e:
            error_msg = f"Error loading segmentation model: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.load_model_btn.setEnabled(True)
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
    
    def run_segmentation(self):
        """Run segmentation on the selected path"""
        if self.segmenter is None:
            napari.utils.notifications.show_info("Please load the segmentation model first")
            return
        
        if len(self.state['paths']) == 0:
            napari.utils.notifications.show_info("Please create a path first")
            return
        
        try:
            # Get the selected path
            path_id = None
            if hasattr(self, 'selected_path_id'):
                path_id = self.selected_path_id
            else:
                # Fallback to the path selected in the list
                selected_items = self.path_list.selectedItems()
                if len(selected_items) == 1:
                    path_id = selected_items[0].data(100)
            
            if path_id is None or path_id not in self.state['paths']:
                napari.utils.notifications.show_info("Please select a path for segmentation")
                self.status_label.setText("Status: No path selected for segmentation")
                return
            
            # Get the path data
            path_data = self.state['paths'][path_id]
            path_name = path_data['name']
            brightest_path = path_data['data']
            
            # Update UI
            self.status_label.setText(f"Status: Running segmentation on {path_name}...")
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
                z_values = [point[0] for point in brightest_path]
                start_frame = int(min(z_values))
                end_frame = int(max(z_values))
            
            print(f"Segmenting path '{path_name}' from frame {start_frame} to {end_frame}")
            print(f"Path has {len(brightest_path)} points")
            
            # Progress callback function
            def update_progress(current, total):
                progress = int((current / total) * 100)
                self.segmentation_progress.setValue(progress)
            
            # Try to run the segmentation
            result_masks = self.segmenter.process_volume(
                image=self.image,
                brightest_path=brightest_path,
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
                segmentation_layer = self.viewer.add_labels(
                    binary_masks,
                    name=seg_layer_name,
                    opacity=0.6,
                )

                segmentation_layer.color = {
                    1: (1.0, 0.0, 1.0),  # magenta
                }
                
                # Store reference in state
                self.state['segmentation_layer'] = segmentation_layer
                
                # Make sure the layer is visible
                if self.state['segmentation_layer'] in self.viewer.layers:
                    self.state['segmentation_layer'].visible = True
                
                # Update the viewer to refresh display
                self.viewer.reset_view()
                
                self.status_label.setText(f"Status: Segmentation complete for {path_name}")
                napari.utils.notifications.show_info(f"Segmentation complete for {path_name}")
                
                # Emit signal that segmentation is completed
                self.segmentation_completed.emit(path_id, seg_layer_name)
            else:
                self.status_label.setText("Status: Segmentation failed. Check console for errors.")
                napari.utils.notifications.show_info("Segmentation failed")
        
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.segmentation_progress.setValue(100)
            self.run_segmentation_btn.setEnabled(True)