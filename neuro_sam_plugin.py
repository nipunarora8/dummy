import napari
import numpy as np
import imageio.v2 as io  # Fixed to avoid deprecation warning
from main_widget import NeuroSAMWidget  # Updated with fast algorithms

import numpy as np

def pad_image_for_patches(image, patch_size=128, pad_value=0):
    """
    Pad the image so that its height and width are multiples of patch_size.
    Handles various image dimensions including stacks of colored images.
    
    Parameters:
    -----------
    image (np.ndarray): Input image array:
        - 2D: (H x W)
        - 3D: (C x H x W) for grayscale stacks or (H x W x C) for colored image
        - 4D: (Z x H x W x C) for stacks of colored images
    patch_size (int): The patch size to pad to, default is 128.
    pad_value (int or tuple): The constant value(s) for padding.
    
    Returns:
    --------
    padded_image (np.ndarray): The padded image.
    padding_amounts (tuple): The amount of padding applied (pad_h, pad_w).
    original_dims (tuple): The original dimensions (h, w).
    """
    # Determine the image format and dimensions
    if image.ndim == 2:
        # 2D grayscale image (H x W)
        h, w = image.shape
        is_color = False
        is_stack = False
    elif image.ndim == 3:
        # This could be either:
        # - A stack of 2D grayscale images (Z x H x W)
        # - A single color image (H x W x C)
        # We'll check the third dimension to decide
        if image.shape[2] <= 4:  # Assuming color channels ≤ 4 (RGB, RGBA)
            # Single color image (H x W x C)
            h, w, c = image.shape
            is_color = True
            is_stack = False
        else:
            # Stack of grayscale images (Z x H x W)
            z, h, w = image.shape
            is_color = False
            is_stack = True
    elif image.ndim == 4:
        # Stack of color images (Z x H x W x C)
        z, h, w, c = image.shape
        is_color = True
        is_stack = True
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")
    
    # Compute necessary padding for height and width
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # Pad the image based on its format
    if not is_stack and not is_color:
        # 2D grayscale image
        padding = ((0, pad_h), (0, pad_w))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif is_stack and not is_color:
        # Stack of grayscale images (Z x H x W)
        padding = ((0, 0), (0, pad_h), (0, pad_w))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif not is_stack and is_color:
        # Single color image (H x W x C)
        padding = ((0, pad_h), (0, pad_w), (0, 0))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    elif is_stack and is_color:
        # Stack of color images (Z x H x W x C)
        padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    
    return padded_image, (pad_h, pad_w), (h, w)

def run_neuro_sam(image=None, image_path=None):
    """
    Launch the NeuroSAM plugin with fast waypoint A* algorithm and optimized tube data generation
    
    Parameters:
    -----------
    image : numpy.ndarray, optional
        3D or higher-dimensional image data. If None, image_path must be provided.
    image_path : str, optional
        Path to image file to load. If None, image must be provided.
    
    Returns:
    --------
    viewer : napari.Viewer
        The napari viewer instance
    """
    # Validate inputs
    if image is None and image_path is None:
        raise ValueError("Either image or image_path must be provided")
    
    # Load image if path provided
    if image is None:
        try:
            image = np.asarray(io.imread(image_path))
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {str(e)}")
    
    # Normalize image to 0-1 range for better visualization
    if image.max() > 1:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

    image, padding_amounts, original_dims = pad_image_for_patches(image, patch_size=128, pad_value=0)   
    
    # Create a viewer
    viewer = napari.Viewer()
    
    # Create and add our widget with enhanced capabilities
    neuro_sam_widget = NeuroSAMWidget(viewer, image)
    viewer.window.add_dock_widget(
        neuro_sam_widget, name="Neuro-SAM Fast", area="right"
    )
    
    # Set initial view
    if image.ndim >= 3:
        # Start with a mid-slice view for 3D+ images
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
    
    return viewer

# For direct execution from command line
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
        viewer = run_neuro_sam(image_path=image_path)
    else:
        # Try to load a default benchmark image
        try:
            image_path = '../DeepD3_Benchmark.tif'
            viewer = run_neuro_sam(image_path=image_path)
        except FileNotFoundError:
            print("Please provide an image path as command line argument")
            print("Usage: python neuro_sam_plugin.py path/to/image.tif")
            sys.exit(1)
    
    napari.run()  # Start the Napari event loop