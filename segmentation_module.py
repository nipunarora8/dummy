import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QCheckBox, QSpinBox
)
from qtpy.QtCore import Signal
from segmentation_model import DendriteSegmenter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from contrasting_color_system import contrasting_color_manager


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

        self.xy_spacing_nm = self.state.get('xy_spacing_nm', 94.0)
        self.z_spacing_nm = self.state.get('z_spacing_nm', 500.0)
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI
        self.setup_ui()

    def update_spacing(self, new_xy_spacing, new_z_spacing):
        """Update spacing for segmentation module"""
        self.xy_spacing_nm = new_xy_spacing
        self.z_spacing_nm = new_z_spacing
        z_scale = new_z_spacing / new_xy_spacing
        print(f"Segmentation: Updated to XY={new_xy_spacing:.1f} nm/pixel, Z={new_z_spacing:.1f} nm/slice")
        print(f"  3D scale ratio (Z/XY): {z_scale:.2f}")
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Model settings
        layout.addWidget(QLabel("<b>Dendrite Segmentation</b>"))
        layout.addWidget(QLabel("1. Load Segmentation Model\n2. Choose the path you want to segment\n3. Click on Run Segmentation to Segment"))
        layout.addWidget(QLabel("<i>Note: Each dendrite gets a unique color with contrasting spine colors</i>"))
        
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
        
        # Export button
        self.export_dendrite_btn = QPushButton("Export Dendrite Masks")
        self.export_dendrite_btn.setFixedHeight(22)
        self.export_dendrite_btn.clicked.connect(self.export_dendrite_masks)
        self.export_dendrite_btn.setEnabled(False)  # Disabled until segmentation exists
        layout.addWidget(self.export_dendrite_btn)
        
        # Progress bar
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setValue(0)
        layout.addWidget(self.segmentation_progress)
        
        # Color info display
        self.color_info_label = QLabel("")
        self.color_info_label.setWordWrap(True)
        layout.addWidget(self.color_info_label)
        
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
                    
                    # Show color info if this path has an assigned color pair
                    color_info = contrasting_color_manager.get_pair_info(path_id)
                    if color_info:
                        self.color_info_label.setText(
                            f"Colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                        )
                    else:
                        self.color_info_label.setText("Colors: Will be assigned during segmentation")
                    
                    # Enable the segmentation button if the model is loaded
                    if self.segmenter is not None:
                        self.run_segmentation_btn.setEnabled(True)
            else:
                # No path selected
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.run_segmentation_btn.setEnabled(False)
                self.color_info_label.setText("")
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
                
                # Get the dendrite color from the contrasting color manager
                dendrite_color = contrasting_color_manager.get_dendrite_color(path_id)
                
                print(f"Adding new segmentation layer: {seg_layer_name}")
                print(f"Result masks shape: {binary_masks.shape}")
                print(f"Result masks type: {binary_masks.dtype}")
                print(f"Result masks min/max: {binary_masks.min()}/{binary_masks.max()}")
                print(f"Using dendrite color: {dendrite_color}")
                
                # Create the segmentation layer using add_image with proper colormap
                # Convert binary masks to float and scale to color range
                color_masks = binary_masks.astype(np.float32)
                color_masks[color_masks > 0] = 1.0  # Ensure binary values
                
                # Add as image layer
                segmentation_layer = self.viewer.add_image(
                    color_masks,
                    name=seg_layer_name,
                    opacity=0.7,
                    blending='additive',
                    colormap='viridis'  # Will be overridden
                )
                
                # Create custom colormap: [transparent, dendrite_color]
                custom_cmap = np.array([
                    [0, 0, 0, 0],  # Transparent for 0 values
                    [dendrite_color[0], dendrite_color[1], dendrite_color[2], 1]  # Color for 1 values
                ])
                
                # Apply the custom colormap
                segmentation_layer.colormap = custom_cmap
                
                # Set contrast limits to ensure proper color mapping
                segmentation_layer.contrast_limits = [0, 1]
                
                print(f"Applied custom colormap: {custom_cmap}")
                print(f"Layer contrast limits: {segmentation_layer.contrast_limits}")
                
                # Store reference in state
                self.state['segmentation_layer'] = segmentation_layer
                
                # Make sure the layer is visible
                segmentation_layer.visible = True
                
                # Update color info display
                color_info = contrasting_color_manager.get_pair_info(path_id)
                if color_info:
                    self.color_info_label.setText(
                        f"Colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                    )
                
                # Enable export button
                self.export_dendrite_btn.setEnabled(True)
                
                self.status_label.setText(f"Status: Segmentation complete for {path_name}")
                napari.utils.notifications.show_info(f"Segmentation complete for {path_name} with contrasting color pair")
                
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
    
    def export_dendrite_masks(self):
        """Export all dendrite segmentation masks"""
        from qtpy.QtWidgets import QFileDialog
        import tifffile
        import os
        from datetime import datetime
        
        try:
            # Check if there are any segmentation layers to export
            dendrite_layers = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Segmentation -' in layer.name:
                    dendrite_layers.append(layer)
            
            if not dendrite_layers:
                napari.utils.notifications.show_info("No dendrite segmentation masks found to export")
                return
            
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Dendrite Masks", ""
            )
            
            if not save_dir:
                return
            
            # Create timestamp for this export session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exported_count = 0
            
            for layer in dendrite_layers:
                try:
                    # Extract path name from layer name
                    path_name = layer.name.replace("Segmentation - ", "").replace(" ", "_")
                    
                    # Get the mask data
                    mask_data = layer.data
                    
                    # Convert to uint8 if needed
                    if mask_data.dtype != np.uint8:
                        mask_data = (mask_data > 0).astype(np.uint8) * 255
                    else:
                        mask_data = mask_data * 255  # Scale to 0-255 range
                    
                    # Create filename
                    filename = f"dendrite_mask_{path_name}_{timestamp}.tif"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save as TIFF
                    tifffile.imwrite(filepath, mask_data)
                    
                    exported_count += 1
                    print(f"Exported dendrite mask: {filepath}")
                    
                except Exception as e:
                    print(f"Error exporting mask for {layer.name}: {str(e)}")
                    continue
            
            if exported_count > 0:
                napari.utils.notifications.show_info(f"Successfully exported {exported_count} dendrite masks to {save_dir}")
                self.status_label.setText(f"Status: Exported {exported_count} dendrite masks")
            else:
                napari.utils.notifications.show_info("No dendrite masks were exported due to errors")
                
        except Exception as e:
            error_msg = f"Error during dendrite mask export: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.status_label.setText(f"Status: {error_msg}")
            print(f"Export error details: {str(e)}")
            import traceback
            traceback.print_exc()