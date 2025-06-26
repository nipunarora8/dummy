import napari
import numpy as np
import imageio as io
from main_widget import NeuroSAMWidget  # Back to original name

def run_neuro_sam(image=None, image_path=None):
    """
    Launch the NeuroSAM plugin with enhanced path tracing algorithm
    
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
    
    # Create a viewer
    viewer = napari.Viewer()
    
    # Create and add our widget (same interface, enhanced backend)
    neuro_sam_widget = NeuroSAMWidget(viewer, image)
    viewer.window.add_dock_widget(
        neuro_sam_widget, name="Neuro-SAM", area="right"
    )
    
    # Set initial view
    if image.ndim >= 3:
        # Start with a mid-slice view for 3D+ images
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
    
    print("\n===== NEURO-SAM: DENDRITE PATH TRACING & SPINE DETECTION =====")
    print("1. Click on the image to set multiple waypoints")
    print("2. Click 'Find Path' to calculate the brightest path")
    print("3. Use the tabs to switch between path management, segmentation, and spine detection")
    print("4. Run segmentation on a path before detecting spines")
    print("5. Use the spine detection tab to find dendritic spines along the segmented path")
    print("=======================================\n")
    
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