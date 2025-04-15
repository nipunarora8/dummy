import napari
import numpy as np
import imageio as io
from brightest_path_widget import BrightestPathWidget

def run_interactive_path_finder(image):
    """
    Launch the interactive brightest path finder with segmentation
    
    Parameters:
    -----------
    image : numpy.ndarray
        3D or higher-dimensional image data
    """
    # Create a viewer
    viewer = napari.Viewer()
    
    # Create and add our widget
    path_widget = BrightestPathWidget(viewer, image)
    viewer.window.add_dock_widget(
        path_widget, name="Brightest Path Finder", area="right"
    )
    
    # Set initial view
    if image.ndim >= 3:
        # Start with a mid-slice view for 3D+ images
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
    
    print("\n===== BRIGHTEST PATH FINDER WITH SEGMENTATION =====")
    print("1. Click on the image to set start and end points")
    print("2. Click 'Find Path' to calculate the path")
    print("3. Go to the Segmentation tab to run dendrite segmentation")
    print("4. Use 'Load Segmentation Model' to initialize the model")
    print("5. Click 'Run Segmentation' to segment along the path")
    print("=======================================\n")
    
    return viewer

# Example usage:
if __name__ == "__main__":
    # Load the image
    image_path = '../DeepD3_Benchmark.tif'
    benchmark = np.asarray(io.imread(image_path))
    
    # Normalize image to 0-1 range for better visualization
    if benchmark.max() > 0:
        benchmark = benchmark.astype(np.float32)
        benchmark = (benchmark - benchmark.min()) / (benchmark.max() - benchmark.min())
    
    # Launch the viewer
    viewer = run_interactive_path_finder(benchmark)
    napari.run()  # Start the Napari event loop