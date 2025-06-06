import numpy as np
import torch
import cv2
from matplotlib.path import Path
from scipy.interpolate import splprep, splev


class DendriteSegmenter:
    """Class for segmenting dendrites from 3D image volumes using SAM2"""
    
    def __init__(self, model_path="checkpoints/sam2.1_hiera_small.pt", config_path="sam2.1_hiera_s.yaml", weights_path="results/samv2_small_2025-03-06-17-13-15/model_22500.torch", device="cpu"):
        """
        Initialize the segmenter.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            config_path: Path to model configuration
            weights_path: Path to trained weights
            device: Device to run the model on (cpu or cuda)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.predictor = None

    def load_model(self):
        """Load the segmentation model with improved error reporting and path handling"""
        try:
            print(f"Loading model from {self.model_path} with config {self.config_path}")
            print(f"Using weights from {self.weights_path}")
            
            # Try importing first to catch import errors
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                print("Successfully imported SAM2 modules")
            except ImportError as ie:
                print(f"Failed to import SAM2 modules: {str(ie)}")
                print("Make sure the SAM2 package is installed and in the Python path")
                return False
            
            # Use bfloat16 for memory efficiency
            torch.autocast(device_type="cpu", dtype=torch.bfloat16).__enter__()

            # Build model and load weights
            print("Building SAM2 model...")
            sam2_model = build_sam2(self.config_path, self.model_path, device=self.device)
            print("Creating SAM2 image predictor...")
            self.predictor = SAM2ImagePredictor(sam2_model)
            print("Loading model weights...")
            self.predictor.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            print("Model loaded successfully")
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def predict_mask(self, img, positive_points, negative_points):
        """
        Predict mask using SAM with positive and negative prompt points
        
        Args:
            img: Input image (2D)
            positive_points: List of (x, y) foreground points
            negative_points: List of (x, y) background points
            
        Returns:
            Binary mask
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Convert to RGB format for SAM
        image_rgb = cv2.merge([img, img, img]).astype(np.float32)
        
        # Prepare points and labels
        points = np.array(positive_points + negative_points, dtype=np.float32)
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)
        
        # Run prediction
        try:
            with torch.no_grad():
                self.predictor.set_image(image_rgb)
                pred, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels
                )
            
            # Return the predicted mask
            return pred[0]
            
        except Exception as e:
            print(f"Error in predict_mask: {e}")
            import traceback
            traceback.print_exc()
            # Return an empty mask as fallback
            return np.zeros_like(img, dtype=np.uint8)
    
    def pad_image_for_patches(self, image, patch_size=128, pad_value=0):
        """
        Pad the image so that its height and width are multiples of patch_size.
        
        Args:
            image: Input image array
            patch_size: The patch size to pad to
            pad_value: The constant value for padding
            
        Returns:
            Padded image, padding dimensions, and original dimensions
        """
        # Get original dimensions
        if image.ndim == 3:
            c, h, w = image.shape
        else:
            h, w = image.shape        
        
        # Compute the necessary padding for height and width
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        padding = ((0, pad_h), (0, pad_w))
        if image.ndim == 3:
            padded_image = np.array([np.pad(i, padding, mode='constant', constant_values=pad_value) for i in image])
        else:
            padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
            
        return padded_image, (pad_h, pad_w), (h, w)
    
    def process_frame(self, frame_idx, image, brightest_path, patch_size=128):
        """
        Process a single frame with patches
        
        Args:
            frame_idx: Index of the frame to process
            image: Input image volume
            brightest_path: List of path points [z, y, x]
            patch_size: Size of patches to process
            
        Returns:
            Predicted mask for the frame
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        height, width = image[frame_idx].shape
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Track if we've processed any patches
        patches_processed = 0
        patches_with_segmentation = 0
        
        # Process each patch
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                # Define patch boundaries
                a, b = i, min(i+patch_size, height)
                c, d = j, min(j+patch_size, width)
                patch = image[frame_idx][a:b, c:d]
                
                # Skip if patch is too small
                if patch.shape[0] < 10 or patch.shape[1] < 10:
                    continue
                    
                # Get original patch dimensions before resizing
                original_height, original_width = patch.shape
                
                # Resize patch to standard size (BEFORE finding points)
                patch = cv2.resize(patch, (patch_size, patch_size), cv2.INTER_LINEAR)
                
                # Find points in the current patch - using the ORIGINAL coordinates
                current_frame_points = []
                for f_idx in range(len(brightest_path)):
                    f = brightest_path[f_idx]
                    if f[0] == frame_idx and a <= f[1] < b and c <= f[2] < d:
                        current_frame_points.append(f)
                        
                # Find points in nearby frames
                nearby_frame_points = []
                frame_range = 3  # Look 3 frames before and after
                for f_idx in range(len(brightest_path)):
                    f = brightest_path[f_idx]
                    if (frame_idx - frame_range <= f[0] <= frame_idx + frame_range and
                        a <= f[1] < b and c <= f[2] < d):
                        nearby_frame_points.append(f)
                        
                # Combine unique points
                all_points = current_frame_points.copy()
                for point in nearby_frame_points:
                    spatial_match = False
                    for current_point in current_frame_points:
                        if point[1] == current_point[1] and point[2] == current_point[2]:
                            spatial_match = True
                            break
                    if not spatial_match:
                        all_points.append(point)
                        
                # Only process if we have enough points
                if len(all_points) >= 3:
                    patches_processed += 1
                    
                    # Define range parameters for negative points
                    min_distance = 5   # Minimum distance from positive points
                    max_distance = 15  # Maximum distance from positive points
                    
                    # Sort points
                    sorted_points = sorted(all_points, key=lambda p: (p[1], p[2]))
                    
                    # Convert coordinates to patch local coordinates
                    # Subtract the patch origin and then apply scaling for resize
                    path_y = [(p[1] - a) * (patch_size / original_height) for p in sorted_points]
                    path_x = [(p[2] - c) * (patch_size / original_width) for p in sorted_points]
                    
                    # Set up containers for SAM points
                    positive_points = []  # Points on the path (foreground)
                    negative_points = []  # Points in the boundary region (background)
                    
                    if len(path_x) >= 3:
                        # Convert to numpy arrays for processing
                        points = np.column_stack([path_x, path_y])
                        
                        # Smooth the path with a spline
                        k = min(3, len(points)-1)
                        try:
                            tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=k)
                            
                            # Generate more points along the spline
                            u_new = np.linspace(0, 1, 100)
                            smooth_x, smooth_y = splev(u_new, tck)
                            
                            # Calculate normals to the path
                            dx = np.gradient(smooth_x)
                            dy = np.gradient(smooth_y)
                            
                            # Normalize vectors and find perpendicular
                            path_length = np.sqrt(dx**2 + dy**2)
                            dx = dx / (path_length + 1e-8)  # Avoid division by zero
                            dy = dy / (path_length + 1e-8)
                            
                            # Perpendicular vectors
                            nx, ny = -dy, dx
                            
                            # Get equidistant points along the path for positive points
                            indices = np.linspace(0, len(smooth_x)-1, 20, dtype=int)
                            for idx in indices:
                                positive_points.append((smooth_x[idx], smooth_y[idx]))
                            
                            # Create distance arrays 
                            inner_buffer_x = smooth_x + nx * min_distance
                            inner_buffer_y = smooth_y + ny * min_distance
                            outer_buffer_x = smooth_x + nx * max_distance
                            outer_buffer_y = smooth_y + ny * max_distance
                            
                            inner_buffer_x_lower = smooth_x - nx * min_distance
                            inner_buffer_y_lower = smooth_y - ny * min_distance
                            outer_buffer_x_lower = smooth_x - nx * max_distance
                            outer_buffer_y_lower = smooth_y - ny * max_distance
                            
                            # Create polygons for the inner and outer boundaries
                            inner_boundary_x = np.concatenate([inner_buffer_x, inner_buffer_x_lower[::-1]])
                            inner_boundary_y = np.concatenate([inner_buffer_y, inner_buffer_y_lower[::-1]])
                            
                            outer_boundary_x = np.concatenate([outer_buffer_x, outer_buffer_x_lower[::-1]])
                            outer_boundary_y = np.concatenate([outer_buffer_y, outer_buffer_y_lower[::-1]])
                            
                            # Create Path objects for checking point containment
                            inner_path = Path(np.column_stack([inner_boundary_x, inner_boundary_y]))
                            outer_path = Path(np.column_stack([outer_boundary_x, outer_boundary_y]))
                            
                            # Find the min/max coordinates of the outer boundary
                            min_x, max_x = np.min(outer_boundary_x), np.max(outer_boundary_x)
                            min_y, max_y = np.min(outer_boundary_y), np.max(outer_boundary_y)
                            
                            # Generate random points within the boundary range
                            neg_count = 0
                            max_attempts = 1000
                            attempts = 0
                            
                            while neg_count < 10 and attempts < max_attempts:
                                rand_x = np.random.uniform(min_x, max_x)
                                rand_y = np.random.uniform(min_y, max_y)
                                if rand_x < 1 or rand_y < 1 or rand_x > patch_size-1 or rand_y > patch_size-1:
                                    attempts += 1
                                    continue
                                
                                # Check if point is between the inner and outer boundaries
                                if outer_path.contains_point((rand_x, rand_y)) and not inner_path.contains_point((rand_x, rand_y)):
                                    negative_points.append((rand_x, rand_y))
                                    neg_count += 1
                                
                                attempts += 1
                                
                        except Exception as e:
                            print(f"Spline interpolation failed: {e}")
                            # Fallback: sample from original points with distance constraints
                            
                            # Select points from original path
                            if len(path_x) <= 20:
                                positive_points = list(zip(path_x, path_y))
                            else:
                                indices = np.linspace(0, len(path_x)-1, 20, dtype=int)
                                positive_points = [(path_x[i], path_y[i]) for i in indices]
                            
                            # Generate negative points
                            for _ in range(10):
                                idx = np.random.randint(0, len(positive_points))
                                px, py = positive_points[idx]
                                
                                angle = np.random.uniform(0, 2*np.pi)
                                radius = np.random.uniform(min_distance, max_distance)
                                nx = px + radius * np.cos(angle)
                                ny = py + radius * np.sin(angle)
                                
                                nx = max(0, min(nx, patch_size-1))
                                ny = max(0, min(ny, patch_size-1))
                                
                                negative_points.append((nx, ny))
                    
                    # Generate prediction mask if we have points
                    if positive_points and negative_points:
                        prediction_mask = self.predict_mask(patch, positive_points, negative_points)
                        
                        # Check if mask is valid and contains segmentation
                        if prediction_mask is not None:
                            # Ensure we have a binary mask (0 or 1)
                            binary_mask = (prediction_mask > 0).astype(np.uint8)
                            
                            # Resize prediction mask back to original patch size if needed
                            if binary_mask.shape != (b-a, d-c):
                                binary_mask = cv2.resize(
                                    binary_mask, (d-c, b-a), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                            
                            # Check if there's actually segmentation in the mask
                            if np.sum(binary_mask) > 0:
                                patches_with_segmentation += 1
                                
                                # Add to the full frame mask
                                pred_mask[a:b, c:d] = binary_mask
                            else:
                                print(f"No segmentation found in patch at ({a},{c})")
                        else:
                            print(f"Prediction mask is None for patch at ({a},{c})")
        
        print(f"Frame {frame_idx}: Processed {patches_processed} patches, {patches_with_segmentation} with segmentation")
        print(f"Frame mask sum: {np.sum(pred_mask)}")
        
        return pred_mask
    
    def process_volume(self, image, brightest_path, start_frame=None, end_frame=None, patch_size=128, progress_callback=None):
        """
        Process a volume of frames
        
        Args:
            image: Input image volume
            brightest_path: List of path points [z, y, x]
            start_frame: First frame to process (default: min z in path)
            end_frame: Last frame to process (default: max z in path)
            patch_size: Size of patches to process
            progress_callback: Optional callback function to report progress
            
        Returns:
            Predicted mask volume
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Convert brightest_path to list if it's a numpy array
        if isinstance(brightest_path, np.ndarray):
            brightest_path = brightest_path.tolist()
            
        # Determine frame range from path if not provided
        if start_frame is None or end_frame is None:
            z_values = [point[0] for point in brightest_path]
            if start_frame is None:
                start_frame = int(min(z_values))
            if end_frame is None:
                end_frame = int(max(z_values))
        
        # Initialize output mask volume
        pred_masks = np.zeros((len(image), image[0].shape[0], image[0].shape[1]), dtype=np.uint8)

        # Process each frame
        total_frames = end_frame - start_frame + 1
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if progress_callback:
                progress_callback(i, total_frames)
            else:
                print(f"Processing frame {frame_idx}/{end_frame} ({i+1}/{total_frames})")
                
            # Process the frame
            frame_mask = self.process_frame(
                frame_idx, image, brightest_path, patch_size=patch_size
            )
            
            # Add the frame mask to the output volume
            pred_masks[frame_idx] = frame_mask
            
        # Check if we have any segmentation
        total_segmentation = np.sum(pred_masks)
        print(f"Total segmentation volume: {total_segmentation} pixels")
        
        if total_segmentation == 0:
            print("WARNING: No segmentation found in any frame!")
            
        return pred_masks