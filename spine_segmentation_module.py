import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFrame, QListWidget, QListWidgetItem,
    QProgressBar, QCheckBox, QSpinBox, QGroupBox
)
from qtpy.QtCore import Signal
import torch
from spine_segmentation_model import SpineSegmenter  # Now with overlapping patches
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from contrasting_color_system import contrasting_color_manager


class SpineSegmentationWidget(QWidget):
    """Widget for segmenting spines using SAMv2 with overlapping patches"""
    
    spine_segmentation_completed = Signal(str, str)  # path_id, layer_name
    
    def __init__(self, viewer, image, state):
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.state = state
        
        # Initialize the spine segmentation model
        self.spine_segmenter = None

        self.xy_spacing_nm = self.state.get('xy_spacing_nm', 94.0)
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Setup UI
        self.setup_ui()

    def update_pixel_spacing(self, new_spacing):
        """Update pixel spacing for spine segmentation module"""
        self.pixel_spacing_nm = new_spacing
        print(f"Overlapping patches spine segmentation: Updated pixel spacing to {new_spacing:.1f} nm/pixel")

    def get_manual_spine_points(self, path_name=None):
        """Get manually clicked spine points from points layers in the viewer."""
        manual_points = []
        original_spine_positions = []
        
        # If we have a path name, check if there's an existing spine layer for this path
        if path_name:
            spine_layer_name = f"Spines - {path_name}"
            for layer in self.viewer.layers:
                if (hasattr(layer, 'name') and layer.name == spine_layer_name and 
                    hasattr(layer, 'data') and len(layer.data) > 0):
                    
                    # Get the original spine positions from our stored data
                    path_id = getattr(self, 'selected_path_id', None)
                    if (path_id and 'spine_data' in self.state and 
                        path_id in self.state['spine_data'] and 
                        'original_positions' in self.state['spine_data'][path_id]):
                        original_spine_positions = self.state['spine_data'][path_id]['original_positions']
                        print(f"Found {len(original_spine_positions)} original spine positions for segmentation")
                    
                    # Check if current layer has more points than originally detected
                    current_points = layer.data
                    if len(current_points) > len(original_spine_positions):
                        # Find new points that weren't in the original detection
                        for point in current_points:
                            point_coords = [point[0], point[1], point[2]]
                            
                            # Check if this point was in the original detection
                            is_original = False
                            for orig_point in original_spine_positions:
                                # Allow small tolerance for floating point differences
                                if (abs(point_coords[0] - orig_point[0]) < 0.1 and 
                                    abs(point_coords[1] - orig_point[1]) < 0.1 and 
                                    abs(point_coords[2] - orig_point[2]) < 0.1):
                                    is_original = True
                                    break
                            
                            if not is_original:
                                manual_points.append(point_coords)
                                print(f"Found new manual spine point for segmentation at {point_coords[0]:.1f}, {point_coords[1]:.1f}, {point_coords[2]:.1f}")
                    
                    break
        
        # Also check other points layers (excluding system layers)
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and hasattr(layer, 'name'):
                # Check if it's a points layer with data
                if (hasattr(layer, 'size') and  # This indicates it's a points layer
                    len(layer.data) > 0 and 
                    layer.data.ndim == 2 and 
                    layer.data.shape[1] >= 3):  # At least 3D coordinates
                    
                    # Skip system layers and the spine layer we already processed
                    skip_layer = False
                    skip_names = ['Path', 'Point Selection', 'Traced Path']
                    if path_name:
                        skip_names.append(f"Spines - {path_name}")
                    
                    for skip_name in skip_names:
                        if skip_name in layer.name:
                            skip_layer = True
                            break
                    
                    if skip_layer:
                        continue
                    
                    # Include points from this layer
                    for point in layer.data:
                        # Ensure we have at least [z, y, x] coordinates
                        if len(point) >= 3:
                            manual_points.append([point[0], point[1], point[2]])
                            print(f"Found manual spine point for segmentation at {point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f} from layer '{layer.name}'")
        
        if manual_points:
            print(f"Total manual spine points found for segmentation: {len(manual_points)}")
        
        return manual_points
    
    def setup_ui(self):
        """Create the UI panel with overlapping patches controls"""
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        
        # Title and instructions
        layout.addWidget(QLabel("<b>Spine Segmentation with Overlapping Patches</b>"))
        layout.addWidget(QLabel("1. Load spine segmentation model\n2. Select a path with detected spines\n3. Optionally add manual spine points\n4. Run spine segmentation"))
        layout.addWidget(QLabel("<i>Note: Uses 50% overlapping patches + circular shape enhancement</i>"))
        
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)
        
        # Model settings section
        model_section = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(2)
        model_layout.setContentsMargins(5, 5, 5, 5)
        
        self.model_path_label = QLabel("SAM2 Model: checkpoints/sam2.1_hiera_small.pt")
        model_layout.addWidget(self.model_path_label)
        
        self.config_path_label = QLabel("Config: sam2.1_hiera_s.yaml")
        model_layout.addWidget(self.config_path_label)
        
        self.weights_path_label = QLabel("Spine Weights: results/samv2_spines_small_2025-06-04-11-08-36/spine_model_58000.torch")
        model_layout.addWidget(self.weights_path_label)
        
        model_section.setLayout(model_layout)
        layout.addWidget(model_section)
        
        # Load model button
        self.load_model_btn = QPushButton("Load Overlapping Patches Segmentation Model")
        self.load_model_btn.setFixedHeight(22)
        self.load_model_btn.clicked.connect(self.load_spine_model)
        layout.addWidget(self.load_model_btn)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Path selection for spine segmentation
        layout.addWidget(QLabel("Select a path with detected spines:"))
        self.path_list = QListWidget()
        self.path_list.setFixedHeight(80)
        self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
        layout.addWidget(self.path_list)
        
        # Overlapping patches parameters
        params_section = QGroupBox("Overlapping Patches Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(2)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Patch size (back to 128x128)
        patch_size_layout = QHBoxLayout()
        patch_size_layout.addWidget(QLabel("Patch Size:"))
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(64, 256)
        self.patch_size_spin.setSingleStep(32)
        self.patch_size_spin.setValue(128)  # Back to proven 128x128
        self.patch_size_spin.setToolTip("Size of overlapping patches (128x128 recommended)")
        patch_size_layout.addWidget(self.patch_size_spin)
        params_layout.addLayout(patch_size_layout)
        
        # Use dendrite mask overlay
        self.use_dendrite_mask_cb = QCheckBox("Use Dendrite Mask Overlay (img[mask] = 0)")
        self.use_dendrite_mask_cb.setChecked(True)
        self.use_dendrite_mask_cb.setToolTip("Suppress dendrite signal using dendrite segmentation mask")
        params_layout.addWidget(self.use_dendrite_mask_cb)
        
        # Circular shape enhancement
        self.enhance_circular_cb = QCheckBox("Enhance Circular Spine Shapes")
        self.enhance_circular_cb.setChecked(True)
        self.enhance_circular_cb.setToolTip("Apply morphological operations to make spines more circular and remove noise")
        params_layout.addWidget(self.enhance_circular_cb)
        
        # Minimum spine size for noise removal
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min Spine Size (pixels):"))
        self.min_spine_size_spin = QSpinBox()
        self.min_spine_size_spin.setRange(5, 50)
        self.min_spine_size_spin.setValue(10)
        self.min_spine_size_spin.setToolTip("Minimum size of spine objects to keep (removes noise)")
        min_size_layout.addWidget(self.min_spine_size_spin)
        params_layout.addLayout(min_size_layout)
        
        # Processing method info
        method_info = QLabel("Method: 50% overlapping patches (stride=64) with adaptive merging\n→ Eliminates boundary artifacts + circular shape enhancement")
        method_info.setWordWrap(True)
        method_info.setStyleSheet("color: #0066cc; font-style: italic;")
        params_layout.addWidget(method_info)
        
        params_section.setLayout(params_layout)
        layout.addWidget(params_section)
        
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator3)
        
        # Run segmentation button
        self.run_segmentation_btn = QPushButton("Run Overlapping Patches Segmentation")
        self.run_segmentation_btn.setFixedHeight(22)
        self.run_segmentation_btn.clicked.connect(self.run_spine_segmentation)
        self.run_segmentation_btn.setEnabled(False)
        layout.addWidget(self.run_segmentation_btn)
        
        # Export button
        self.export_spine_btn = QPushButton("Export Spine Masks")
        self.export_spine_btn.setFixedHeight(22)
        self.export_spine_btn.clicked.connect(self.export_spine_masks)
        self.export_spine_btn.setEnabled(False)  # Disabled until segmentation exists
        layout.addWidget(self.export_spine_btn)
        
        # Progress bar
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setValue(0)
        layout.addWidget(self.segmentation_progress)
        
        # Color info display
        self.color_info_label = QLabel("")
        self.color_info_label.setWordWrap(True)
        layout.addWidget(self.color_info_label)
        
        # Status and results
        self.results_label = QLabel("Results: No spine segmentation performed yet")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
        self.status_label = QLabel("Status: Model not loaded")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
    
    def load_spine_model(self):
        """Load the spine segmentation model with overlapping patches"""
        try:
            self.status_label.setText("Status: Loading overlapping patches spine segmentation model...")
            self.load_model_btn.setEnabled(False)
            
            # Initialize segmenter if not already done
            if self.spine_segmenter is None:
                self.spine_segmenter = SpineSegmenter(
                    model_path="../Fine-Tune-SAMv2/checkpoints/sam2.1_hiera_small.pt",
                    config_path="sam2.1_hiera_s.yaml",
                    weights_path="../Fine-Tune-SAMv2/results/samv2_spines_small_2025-06-04-11-08-36/spine_model_58000.torch",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            # Load the model
            success = self.spine_segmenter.load_model()
            
            if success:
                self.status_label.setText("Status: Overlapping patches spine segmentation model loaded successfully!")
                self.update_path_list()  # Update available paths
                napari.utils.notifications.show_info("Overlapping patches spine segmentation model loaded successfully")
            else:
                self.status_label.setText("Status: Failed to load overlapping patches spine segmentation model")
                self.load_model_btn.setEnabled(True)
                napari.utils.notifications.show_info("Failed to load overlapping patches spine segmentation model")
                
        except Exception as e:
            error_msg = f"Error loading overlapping patches spine segmentation model: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.load_model_btn.setEnabled(True)
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
    
    def update_path_list(self):
        """Update the path list with paths that have detected spines"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            self.path_list.clear()
            
            # Only show paths that have spine detection completed
            for path_id, path_data in self.state['paths'].items():
                path_name = path_data['name']
                spine_layer_name = f"Spines - {path_name}"
                
                # Check if this path has spine detection
                has_spines = False
                if path_id in self.state.get('spine_layers', {}):
                    has_spines = True
                else:
                    # Also check by layer name
                    for layer in self.viewer.layers:
                        if layer.name == spine_layer_name:
                            has_spines = True
                            break
                
                if has_spines:
                    item = QListWidgetItem(path_name)
                    item.setData(100, path_id)
                    self.path_list.addItem(item)
            
            # Enable segmentation button if model is loaded and a path is available
            self.run_segmentation_btn.setEnabled(
                self.spine_segmenter is not None and 
                self.path_list.count() > 0
            )
            
            if self.path_list.count() == 0:
                self.status_label.setText("Status: No paths with spine detection found")
            
        except Exception as e:
            napari.utils.notifications.show_info(f"Error updating spine segmentation path list: {str(e)}")
            self.status_label.setText(f"Status: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_path_selection_changed(self):
        """Handle when path selection changes"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) == 1:
                path_id = selected_items[0].data(100)
                if path_id in self.state['paths']:
                    self.selected_path_id = path_id
                    path_name = self.state['paths'][path_id]['name']
                    
                    # Check if this path already has spine segmentation
                    spine_seg_layer_name = f"Spine Segmentation - {path_name}"
                    has_spine_segmentation = False
                    for layer in self.viewer.layers:
                        if layer.name == spine_seg_layer_name:
                            has_spine_segmentation = True
                            break
                    
                    # Check for manual points
                    manual_points = self.get_manual_spine_points(path_name)
                    manual_info = f" (+{len(manual_points)} manual points)" if manual_points else ""
                    
                    # Check if dendrite segmentation exists
                    dendrite_seg_layer_name = f"Segmentation - {path_name}"
                    has_dendrite_segmentation = False
                    for layer in self.viewer.layers:
                        if layer.name == dendrite_seg_layer_name:
                            has_dendrite_segmentation = True
                            break
                    
                    if has_spine_segmentation:
                        status_msg = f"Status: Path '{path_name}' already has overlapping patches spine segmentation{manual_info}"
                    else:
                        status_msg = f"Status: Path '{path_name}' selected for overlapping patches spine segmentation{manual_info}"
                        if not has_dendrite_segmentation:
                            status_msg += " (Note: No dendrite segmentation found - will skip mask overlay)"
                    
                    self.status_label.setText(status_msg)
                    
                    # Show color info for this path
                    color_info = contrasting_color_manager.get_pair_info(path_id)
                    if color_info:
                        self.color_info_label.setText(
                            f"Contrasting colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                        )
                    else:
                        self.color_info_label.setText("Note: Segment the dendrite first to assign color pair")
                    
                    # Enable segmentation button if model is loaded
                    if self.spine_segmenter is not None:
                        self.run_segmentation_btn.setEnabled(True)
            else:
                if hasattr(self, 'selected_path_id'):
                    delattr(self, 'selected_path_id')
                self.run_segmentation_btn.setEnabled(False)
                self.color_info_label.setText("")
                
        except Exception as e:
            napari.utils.notifications.show_info(f"Error handling overlapping patches spine segmentation path selection: {str(e)}")
        finally:
            self.handling_event = False
    
    def run_spine_segmentation(self):
        """Run spine segmentation with overlapping patches"""
        if self.spine_segmenter is None:
            napari.utils.notifications.show_info("Please load the spine segmentation model first")
            return
        
        if not hasattr(self, 'selected_path_id'):
            napari.utils.notifications.show_info("Please select a path for spine segmentation")
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
            
            # Get manually clicked spine points
            manual_spine_points = self.get_manual_spine_points(path_name)
            
            # Get segmentation parameters
            use_dendrite_mask = self.use_dendrite_mask_cb.isChecked()
            enhance_circular = self.enhance_circular_cb.isChecked()
            min_spine_size = self.min_spine_size_spin.value()
            patch_size = self.patch_size_spin.value()
            
            # Update UI
            enhancement_info = [f"{patch_size}x{patch_size} overlapping patches (50%)"]
            if use_dendrite_mask:
                enhancement_info.append("dendrite suppression")
            if enhance_circular:
                enhancement_info.append("circular shape enhancement")
            
            enhancement_str = " + ".join(enhancement_info)
            manual_info = f" (including {len(manual_spine_points)} manual points)" if manual_spine_points else ""
            
            self.status_label.setText(f"Status: Running overlapping patches spine segmentation on {path_name}{manual_info} with {enhancement_str}...")
            self.segmentation_progress.setValue(10)
            self.run_segmentation_btn.setEnabled(False)
            
            # Get spine positions for this path
            spine_positions = None
            
            # First try to get from spine_data (enhanced detection)
            if ('spine_data' in self.state and 
                path_id in self.state['spine_data'] and 
                'original_positions' in self.state['spine_data'][path_id]):
                spine_positions = self.state['spine_data'][path_id]['original_positions']
                print(f"Using enhanced spine detection data: {len(spine_positions)} spines")
            
            # Fallback to spine layer data
            elif path_id in self.state.get('spine_layers', {}):
                spine_layer = self.state['spine_layers'][path_id]
                if len(spine_layer.data) > 0:
                    spine_positions = spine_layer.data
                    print(f"Using spine layer data: {len(spine_positions)} spines")
            
            # Last resort: look for spine layer by name
            else:
                spine_layer_name = f"Spines - {path_name}"
                for layer in self.viewer.layers:
                    if layer.name == spine_layer_name and len(layer.data) > 0:
                        spine_positions = layer.data
                        print(f"Using spine layer by name: {len(spine_positions)} spines")
                        break
            
            # Add manual spine points to the spine positions
            if manual_spine_points and len(manual_spine_points) > 0:
                if spine_positions is not None:
                    # Convert to numpy array if needed
                    if not isinstance(spine_positions, np.ndarray):
                        spine_positions = np.array(spine_positions)
                    
                    # Add manual points to existing spine positions
                    manual_array = np.array(manual_spine_points)
                    spine_positions = np.vstack([spine_positions, manual_array])
                    print(f"Added {len(manual_spine_points)} manual points to {len(spine_positions) - len(manual_spine_points)} detected spines")
                else:
                    # Only manual points available
                    spine_positions = np.array(manual_spine_points)
                    print(f"Using only {len(manual_spine_points)} manual spine points")
            
            if spine_positions is None or len(spine_positions) == 0:
                napari.utils.notifications.show_info(f"No spine positions found for {path_name}")
                self.status_label.setText(f"Status: No spine positions found for {path_name}")
                self.run_segmentation_btn.setEnabled(True)
                return
            
            # Convert to numpy array if needed
            if not isinstance(spine_positions, np.ndarray):
                spine_positions = np.array(spine_positions)
            
            # Get dendrite segmentation mask if requested
            dendrite_mask = None
            if use_dendrite_mask:
                dendrite_seg_layer_name = f"Segmentation - {path_name}"
                for layer in self.viewer.layers:
                    if layer.name == dendrite_seg_layer_name:
                        dendrite_mask = layer.data
                        print(f"Found dendrite segmentation mask: {dendrite_mask.shape}")
                        print(f"Dendrite mask coverage: {np.sum(dendrite_mask > 0)} pixels")
                        break
                
                if dendrite_mask is None:
                    print("Warning: Dendrite mask overlay requested but no dendrite segmentation found")
                    napari.utils.notifications.show_info(f"Warning: No dendrite segmentation found for {path_name}, skipping mask overlay")
            
            self.segmentation_progress.setValue(20)
            
            print(f"Running overlapping patches spine segmentation on {path_name} with {len(spine_positions)} spine positions")
            print(f"Parameters: patch_size={patch_size}x{patch_size}, overlap=50% (stride={patch_size//2})")
            print(f"Dendrite mask: {dendrite_mask is not None}, Circular enhancement: {enhance_circular}")
            print(f"Min spine size: {min_spine_size} pixels")
            
            # Progress callback
            def update_progress(current, total):
                progress = int(20 + (current / total) * 70)  # 20-90%
                self.segmentation_progress.setValue(progress)
            
            # Run overlapping patches spine segmentation
            spine_masks = self.spine_segmenter.process_volume_spines(
                image=self.image,
                spine_positions=spine_positions,
                dendrite_mask=dendrite_mask,  # Pass dendrite mask for suppression
                patch_size=patch_size,        # Use user-specified patch size
                progress_callback=update_progress
            )
            
            self.segmentation_progress.setValue(90)
            
            # Process results
            if spine_masks is not None:
                # Ensure masks are binary
                binary_spine_masks = (spine_masks > 0).astype(np.uint8)
                
                # Create overlapping patches spine segmentation layer
                spine_seg_layer_name = f"Spine Segmentation - {path_name}"
                
                # Remove existing layer if it exists
                for layer in list(self.viewer.layers):
                    if layer.name == spine_seg_layer_name:
                        self.viewer.layers.remove(layer)
                        break
                
                # Get the contrasting spine color for this path
                spine_neon_color = contrasting_color_manager.get_spine_color(path_id)
                
                print(f"Adding overlapping patches spine segmentation layer: {spine_seg_layer_name}")
                print(f"Spine masks shape: {binary_spine_masks.shape}")
                print(f"Spine masks type: {binary_spine_masks.dtype}")
                print(f"Spine masks min/max: {binary_spine_masks.min()}/{binary_spine_masks.max()}")
                print(f"Total segmented pixels: {np.sum(binary_spine_masks)}")
                print(f"Using contrasting neon color: {spine_neon_color}")
                
                # Create the spine segmentation layer using add_image with proper colormap
                # Convert binary masks to float and scale to color range
                color_spine_masks = binary_spine_masks.astype(np.float32)
                color_spine_masks[color_spine_masks > 0] = 1.0  # Ensure binary values
                
                # Add as image layer
                spine_segmentation_layer = self.viewer.add_image(
                    color_spine_masks,
                    name=spine_seg_layer_name,
                    opacity=0.8,  # Slightly higher opacity for spines
                    blending='additive',
                    colormap='viridis'  # Will be overridden
                )
                
                # Create custom neon colormap: [transparent, neon_color]
                custom_neon_cmap = np.array([
                    [0, 0, 0, 0],  # Transparent for 0 values
                    [spine_neon_color[0], spine_neon_color[1], spine_neon_color[2], 1]  # Neon color for 1 values
                ])
                
                # Apply the custom neon colormap
                spine_segmentation_layer.colormap = custom_neon_cmap
                
                # Set contrast limits to ensure proper color mapping
                spine_segmentation_layer.contrast_limits = [0, 1]
                
                print(f"Applied custom neon colormap: {custom_neon_cmap}")
                print(f"Spine layer contrast limits: {spine_segmentation_layer.contrast_limits}")
                
                # Store reference in state
                if 'spine_segmentation_layers' not in self.state:
                    self.state['spine_segmentation_layers'] = {}
                self.state['spine_segmentation_layers'][path_id] = spine_segmentation_layer
                
                # Make sure the layer is visible
                spine_segmentation_layer.visible = True
                
                # Update color info display
                color_info = contrasting_color_manager.get_pair_info(path_id)
                if color_info:
                    self.color_info_label.setText(
                        f"Contrasting colors: Dendrite {color_info['dendrite_hex']} -> Spine {color_info['spine_hex']}"
                    )
                
                # Enable export button
                self.export_spine_btn.setEnabled(True)
                
                # Update UI with overlapping patches information
                total_pixels = np.sum(binary_spine_masks)
                manual_count = len(manual_spine_points) if manual_spine_points else 0
                detected_count = len(spine_positions) - manual_count
                
                result_text = f"Results: Overlapping patches spine segmentation completed - {total_pixels} pixels segmented"
                if manual_count > 0:
                    result_text += f"\nUsed {detected_count} detected + {manual_count} manual spine positions"
                else:
                    result_text += f"\nUsed {detected_count} detected spine positions"
                
                result_text += f"\nMethod: {enhancement_str}"
                result_text += f"\nOverlap: 50% (stride={patch_size//2})"
                if dendrite_mask is not None:
                    result_text += f"\nDendrite suppression: {np.sum(dendrite_mask > 0)} pixels masked"
                
                self.results_label.setText(result_text)
                self.status_label.setText(f"Status: Overlapping patches spine segmentation completed for {path_name}")
                
                # Create comprehensive notification
                notification_msg = f"Overlapping patches spine segmentation completed for {path_name}"
                if manual_count > 0:
                    notification_msg += f" (including {manual_count} manual points)"
                notification_msg += f" with circular enhancement and noise removal"
                
                napari.utils.notifications.show_info(notification_msg)
                
                # Emit signal that spine segmentation is completed
                self.spine_segmentation_completed.emit(path_id, spine_seg_layer_name)
            else:
                self.results_label.setText("Results: Overlapping patches spine segmentation failed")
                self.status_label.setText("Status: Overlapping patches spine segmentation failed. Check console for errors.")
                napari.utils.notifications.show_info("Overlapping patches spine segmentation failed")
        
        except Exception as e:
            error_msg = f"Error during overlapping patches spine segmentation: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            self.results_label.setText("Results: Error during overlapping patches spine segmentation")
            napari.utils.notifications.show_info(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.segmentation_progress.setValue(100)
            self.run_segmentation_btn.setEnabled(True)
    
    def export_spine_masks(self):
        """Export all spine segmentation masks"""
        from qtpy.QtWidgets import QFileDialog
        import tifffile
        import os
        from datetime import datetime
        
        try:
            # Check if there are any spine segmentation layers to export
            spine_layers = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and 'Spine Segmentation -' in layer.name:
                    spine_layers.append(layer)
            
            if not spine_layers:
                napari.utils.notifications.show_info("No spine segmentation masks found to export")
                return
            
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Overlapping Patches Spine Masks", ""
            )
            
            if not save_dir:
                return
            
            # Create timestamp for this export session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exported_count = 0
            
            for layer in spine_layers:
                try:
                    # Extract path name from layer name
                    path_name = layer.name.replace("Spine Segmentation - ", "").replace(" ", "_")
                    
                    # Get the mask data
                    mask_data = layer.data
                    
                    # Convert to uint8 if needed
                    if mask_data.dtype != np.uint8:
                        mask_data = (mask_data > 0).astype(np.uint8) * 255
                    else:
                        mask_data = mask_data * 255  # Scale to 0-255 range
                    
                    # Create filename
                    filename = f"overlapping_patches_spine_mask_{path_name}_{timestamp}.tif"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save as TIFF
                    tifffile.imwrite(filepath, mask_data)
                    
                    exported_count += 1
                    print(f"Exported overlapping patches spine mask: {filepath}")
                    
                except Exception as e:
                    print(f"Error exporting mask for {layer.name}: {str(e)}")
                    continue
            
            if exported_count > 0:
                napari.utils.notifications.show_info(f"Successfully exported {exported_count} overlapping patches spine masks to {save_dir}")
                self.status_label.setText(f"Status: Exported {exported_count} overlapping patches spine masks")
            else:
                napari.utils.notifications.show_info("No spine masks were exported due to errors")
                
        except Exception as e:
            error_msg = f"Error during spine mask export: {str(e)}"
            napari.utils.notifications.show_info(error_msg)
            self.status_label.setText(f"Status: {error_msg}")
            print(f"Export error details: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def enable_for_path(self, path_id):
        """Enable spine segmentation for a specific path"""
        self.update_path_list()
        
        # Select the path in the list
        for i in range(self.path_list.count()):
            item = self.path_list.item(i)
            if item.data(100) == path_id:
                self.path_list.setCurrentItem(item)
                break