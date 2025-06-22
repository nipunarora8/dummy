import numpy as np
import torch
import cv2
from skimage.filters import threshold_otsu


class SpineSegmenter:
    """Class for segmenting individual spines using SAM2"""
    
    def __init__(self, model_path="checkpoints/sam2.1_hiera_small.pt", 
                 config_path="sam2.1_hiera_s.yaml", 
                 weights_path="results/samv2_spines_small_2025-06-04-11-08-36/spine_model_58000.torch", 
                 device="cuda"):
        """
        Initialize the spine segmenter.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            config_path: Path to model configuration
            weights_path: Path to trained spine weights
            device: Device to run the model on
        """
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.predictor = None

    def load_model(self):
        """Load the spine segmentation model"""
        try:
            print(f"Loading spine segmentation model from {self.model_path}")
            
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Use bfloat16 for memory efficiency
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

            # Build model and load weights
            sam2_model = build_sam2(self.config_path, self.model_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.predictor.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            
            print("Spine segmentation model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading spine segmentation model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def detect_spine_centers_from_array(self, mask, min_distance=3):
        """Detect spine centers from binary mask (adapted from your original code)"""
        from scipy.ndimage import label, center_of_mass
        from scipy.spatial.distance import pdist, squareform
        
        # Label connected components
        labeled_mask, num_features = label(mask)
        
        if num_features == 0:
            return [], []
        
        # Get centers of mass
        centers = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
        centers = np.array(centers)
        
        if len(centers) == 0:
            return [], []
        
        # Filter centers based on minimum distance
        if len(centers) > 1:
            distances = pdist(centers)
            distance_matrix = squareform(distances)
            
            # Keep only centers that are far enough apart
            keep_indices = []
            for i in range(len(centers)):
                if i == 0:
                    keep_indices.append(i)
                else:
                    min_dist_to_kept = min([distance_matrix[i, j] for j in keep_indices])
                    if min_dist_to_kept >= min_distance:
                        keep_indices.append(i)
            
            centers = centers[keep_indices]
        
        # Convert to (x, y) format and create labels
        spine_points = [(center[1], center[0]) for center in centers]  # (x, y)
        spine_labels = [1] * len(spine_points)  # All positive points
        
        return spine_points, spine_labels

    def predict_spine_mask(self, image, spine_points):
        """
        Predict spine mask using SAM with spine center points as prompts
        
        Args:
            image: Input image (2D)
            spine_points: List of (x, y) spine center coordinates
            
        Returns:
            Binary mask of segmented spines
        """
        if self.predictor is None:
            print("Spine segmentation model not loaded. Call load_model() first.")
            return None
            
        if not spine_points:
            return np.zeros_like(image, dtype=np.uint8)
            
        # Convert to RGB format for SAM
        image_rgb = cv2.merge([image, image, image]).astype(np.float32)
        
        # Prepare points and labels (all spine centers are positive)
        points = np.array(spine_points, dtype=np.float32)
        labels = np.array([1] * len(spine_points), dtype=np.int32)
        print(points)
        
        try:
            with torch.no_grad():
                self.predictor.set_image(image_rgb)
                pred, _, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels
                )
            
            return pred[0].astype(np.uint8)
            
        except Exception as e:
            print(f"Error in spine prediction: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(image, dtype=np.uint8)

    def pad_image_for_patches(self, image, patch_size=128, pad_value=0):
        """Pad image to be divisible by patch_size"""
        h, w = image.shape[:2]
        
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if image.ndim == 3:
            padding = ((0, pad_h), (0, pad_w), (0, 0))
        else:
            padding = ((0, pad_h), (0, pad_w))
        
        padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
        return padded_image, (pad_h, pad_w), (h, w)

    def process_frame_patches(self, frame_idx, image, spine_positions, patch_size=128):
        """
        Process a single frame with patches for spine segmentation
        
        Args:
            frame_idx: Index of the frame to process
            image: Input image volume
            spine_positions: List of spine positions for this frame
            patch_size: Size of patches to process
            
        Returns:
            Predicted spine mask for the frame
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
            
        # Get frame-specific spine positions
        frame_spines = [pos for pos in spine_positions if int(pos[0]) == frame_idx]
        
        if not frame_spines:
            return np.zeros_like(image[frame_idx], dtype=np.uint8)
            
        height, width = image[frame_idx].shape
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get frame image
        frame_img = (image[frame_idx] - image[frame_idx].min()) / (image[frame_idx].max() - image[frame_idx].min())
        
        # Pad image
        padded_img, (pad_h, pad_w), (orig_h, orig_w) = self.pad_image_for_patches(frame_img, patch_size)
        padded_mask = np.zeros_like(padded_img, dtype=np.uint8)
        
        # Get image threshold for processing decisions
        try:
            otsu_threshold = threshold_otsu(frame_img)
        except:
            otsu_threshold = 0.1
        
        # Process each patch
        for i in range(0, padded_img.shape[0], patch_size):
            for j in range(0, padded_img.shape[1], patch_size):
                # Extract patch
                patch = padded_img[i:i+patch_size, j:j+patch_size]
                
                # Skip if patch is too small or too dark
                if patch.shape[0] < 10 or patch.shape[1] < 10:
                    continue
                
                try:
                    patch_threshold = threshold_otsu(patch)
                    if patch_threshold < otsu_threshold / 2:
                        continue
                except:
                    continue
                
                # Find spine positions in this patch
                patch_spine_points = []
                for spine_pos in frame_spines:
                    # Convert 3D position to 2D patch coordinates
                    y, x = spine_pos[1], spine_pos[2]
                    
                    # Check if spine is in this patch (original coordinates)
                    if i <= y < i + patch_size and j <= x < j + patch_size:
                        # Convert to patch-local coordinates
                        local_x = x - j
                        local_y = y - i
                        patch_spine_points.append((local_x, local_y))
                
                # Process patch if it contains spines
                if patch_spine_points:
                    # Resize patch to standard size if needed
                    if patch.shape != (patch_size, patch_size):
                        original_shape = patch.shape
                        patch_resized = cv2.resize(patch, (patch_size, patch_size), cv2.INTER_LINEAR)
                        
                        # Scale spine coordinates
                        scaled_spine_points = []
                        for px, py in patch_spine_points:
                            scaled_x = px * (patch_size / original_shape[1])
                            scaled_y = py * (patch_size / original_shape[0])
                            scaled_spine_points.append((scaled_x, scaled_y))
                        
                        patch_spine_points = scaled_spine_points
                        patch = patch_resized
                    
                    # Predict spine mask for this patch
                    patch_mask = self.predict_spine_mask(patch, patch_spine_points)
                    
                    if patch_mask is not None and np.sum(patch_mask) > 0:
                        # Resize back to original patch size if needed
                        if patch_mask.shape != (i + patch_size - i, j + patch_size - j):
                            target_h = min(patch_size, padded_img.shape[0] - i)
                            target_w = min(patch_size, padded_img.shape[1] - j)
                            patch_mask = cv2.resize(
                                patch_mask, (target_w, target_h), 
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Add to full mask
                        end_i = min(i + patch_mask.shape[0], padded_mask.shape[0])
                        end_j = min(j + patch_mask.shape[1], padded_mask.shape[1])
                        padded_mask[i:end_i, j:end_j] = patch_mask[:end_i-i, :end_j-j]
        
        # Remove padding and return original size mask
        pred_mask = padded_mask[:orig_h, :orig_w]
        
        return pred_mask

    def process_volume_spines(self, image, spine_positions, patch_size=128, progress_callback=None):
        """
        Process a volume for spine segmentation
        
        Args:
            image: Input image volume
            spine_positions: List of spine positions [z, y, x]
            patch_size: Size of patches to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Predicted spine mask volume
        """
        if self.predictor is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        # Convert spine positions to numpy array if needed
        if isinstance(spine_positions, list):
            spine_positions = np.array(spine_positions)
        
        # Determine frame range from spine positions
        if len(spine_positions) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        z_values = spine_positions[:, 0].astype(int)
        start_frame = max(0, int(np.min(z_values)))
        end_frame = min(len(image) - 1, int(np.max(z_values)))
        
        # Initialize output mask volume
        pred_masks = np.zeros_like(image, dtype=np.uint8)
        
        total_frames = end_frame - start_frame + 1
        
        # Process each frame
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if progress_callback:
                progress_callback(i, total_frames)
            else:
                print(f"Processing spine segmentation frame {frame_idx}/{end_frame} ({i+1}/{total_frames})")
            
            # Process the frame
            frame_mask = self.process_frame_patches(
                frame_idx, image, spine_positions, patch_size=patch_size
            )
            
            if frame_mask is not None:
                pred_masks[frame_idx] = frame_mask
        
        total_segmentation = np.sum(pred_masks)
        print(f"Total spine segmentation volume: {total_segmentation} pixels")
        
        return pred_masks