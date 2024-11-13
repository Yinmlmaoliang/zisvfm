import torch
import cv2
import numpy as np
from itertools import combinations

def extract_largest_component(binary_masks: torch.Tensor) -> torch.Tensor:
    """
    Extracts the largest connected component from each binary mask in a batch.
    
    Args:
        binary_masks (torch.Tensor): Batch of binary masks with shape (N, H, W)
    
    Returns:
        torch.Tensor: Processed masks containing only the largest component per mask
    """
    processed_masks = torch.zeros_like(binary_masks)
    
    for idx in range(binary_masks.shape[0]):
        # Convert mask to numpy for OpenCV processing
        current_mask = binary_masks[idx].cpu().numpy()
        
        # Analyze connected components
        components_data = cv2.connectedComponentsWithStats(
            current_mask.astype('uint8'),
            connectivity=8
        )
        num_components, labels, stats, _ = components_data
        
        if num_components >= 2:
            # Get index of largest component (excluding background)
            largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Create mask with only largest component
            largest_component = (labels == largest_component_idx).astype('uint8')
            processed_masks[idx] = torch.from_numpy(largest_component)
        else:
            # Keep original mask if no components found
            processed_masks[idx] = binary_masks[idx]
    
    return processed_masks


def calculate_bounding_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates bounding boxes for the largest component in each binary mask.
    
    Args:
        binary_masks (torch.Tensor): Batch of binary masks with shape (N, H, W)
        
    Returns:
        torch.Tensor: Bounding boxes with shape (N, 4) in format [x1, y1, x2, y2]
    """
    batch_size, height, width = masks.shape
    bounding_boxes = []
    
    for idx in range(batch_size):
        # Convert mask to numpy for OpenCV processing
        current_mask = masks[idx].cpu().numpy().astype('uint8')
        
        # Analyze connected components
        components_data = cv2.connectedComponentsWithStats(
            current_mask,
            connectivity=8
        )
        _, _, stats, _ = components_data
        
        # Extract or default bounding box coordinates
        if len(stats) > 1:
            x, y, w, h, _ = stats[1]  # stats[1] contains largest component data
        else:
            x, y, w, h = 0, 0, width, height
            
        # Calculate corner coordinates
        x2, y2 = x + w, y + h
        
        # Ensure coordinates are within image boundaries
        box_coords = [
            max(0, float(x)),
            max(0, float(y)),
            min(width, float(x2)),
            min(height, float(y2))
        ]
        bounding_boxes.append(box_coords)
    
    # Convert to tensor on same device as input
    return torch.tensor(bounding_boxes, device=masks.device)

def process_overlapping_masks(mask_data, overlap_threshold=0.90, max_combination_size=3):
    """
    Process masks to ensure independence by removing union masks and handling overlapping regions
    while preserving corresponding bounding boxes.
    
    Args:
        mask_data (dict): Dictionary containing mask and box tensors.
            - "masks": torch.Tensor of shape [N, H, W], binary masks
            - "boxes": torch.Tensor of shape [N, 4], bounding boxes in [x1, y1, x2, y2] format
        overlap_threshold (float): IoU threshold for identifying union masks (default: 0.90)
        max_combination_size (int): Maximum size of mask combinations to check (default: 3)
    
    Returns:
        dict: Dictionary containing processed independent masks and boxes
            - "masks": torch.Tensor of shape [M, H, W]
            - "boxes": torch.Tensor of shape [M, 4]
    """
    # Extract input tensors
    input_masks = mask_data["masks"]  # [N, H, W]
    input_boxes = mask_data["boxes"]  # [N, 4]
    num_masks, height, width = input_masks.shape
    device = input_masks.device

    # Convert masks to boolean type and calculate areas
    binary_masks = input_masks.bool()
    mask_areas = binary_masks.view(num_masks, -1).sum(dim=1)  # [N]

    # Step 1: Identify and remove union masks
    union_indices = set()
    
    for current_idx in range(num_masks):
        if current_idx in union_indices:
            continue
            
        # Try different combination sizes
        for combo_size in range(2, max_combination_size + 1):
            # Get valid mask indices for combinations
            available_indices = [idx for idx in range(num_masks) 
                               if idx != current_idx and idx not in union_indices]
            
            if len(available_indices) < combo_size:
                continue
                
            # Check all possible combinations
            for mask_combo in combinations(available_indices, combo_size):
                # Compute union of combined masks
                union_mask = binary_masks[list(mask_combo)].any(dim=0)
                
                # Calculate IoU with current mask
                intersection = (binary_masks[current_idx] & union_mask).sum().item()
                union = (binary_masks[current_idx] | union_mask).sum().item()
                iou = intersection / union if union > 0 else 0
                
                if iou >= overlap_threshold:
                    union_indices.add(current_idx)
                    break
                    
            if current_idx in union_indices:
                break

    # Remove identified union masks
    if union_indices:
        mask_indices = torch.arange(num_masks, device=device)
        union_tensor = torch.tensor(list(union_indices), device=device)
        valid_mask = ~torch.isin(mask_indices, union_tensor)
        
        processed_masks = binary_masks[valid_mask]
        processed_boxes = input_boxes[valid_mask]
        processed_areas = mask_areas[valid_mask]
    else:
        processed_masks = binary_masks
        processed_boxes = input_boxes
        processed_areas = mask_areas

    # Step 2: Sort masks by area
    _, sort_indices = torch.sort(processed_areas)
    sorted_masks = processed_masks[sort_indices]
    sorted_boxes = processed_boxes[sort_indices]

    # Step 3: Ensure mask independence
    final_masks = []
    final_boxes = []
    occupied_pixels = torch.zeros((height, width), dtype=torch.bool, device=device)

    for idx in range(len(sorted_masks)):
        current_mask = sorted_masks[idx]
        current_box = sorted_boxes[idx]
        
        # Remove overlapping regions
        independent_region = current_mask & ~occupied_pixels
        
        if independent_region.any():
            final_masks.append(independent_region.unsqueeze(0))
            final_boxes.append(current_box.unsqueeze(0))
            occupied_pixels |= independent_region

    # Prepare output tensors
    if final_masks:
        result_masks = torch.cat(final_masks, dim=0)
        result_boxes = torch.cat(final_boxes, dim=0)
    else:
        result_masks = torch.empty((0, height, width), dtype=torch.bool, device=device)
        result_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)

    return {
        "masks": result_masks,
        "boxes": result_boxes
    }

def calculate_patch_assignments(input_data, image_size=(480, 640), patch_size=14, 
                              use_mask=True, coverage_threshold=0.1):
    """
    Calculate patch assignments for image regions based on either masks or bounding boxes.
    Each patch is assigned to at most one object based on the highest coverage ratio.
    
    Args:
        input_data (dict): Dictionary containing object data:
            - 'masks': torch.Tensor [num_objects, height, width] - Binary object masks
            - 'boxes': torch.Tensor [num_objects, 4] - Bounding boxes [x1, y1, x2, y2]
        image_size (tuple): Original image dimensions (height, width)
        patch_size (int): Size of each square patch
        use_mask (bool): Whether to use masks (True) or bounding boxes (False)
        coverage_threshold (float): Minimum ratio of mask pixels per patch for assignment
    
    Returns:
        dict: Mapping of object indices to their assigned patch indices.
              None value indicates no patches assigned to that object.
    """
    # Calculate padding to ensure image dimensions are divisible by patch_size
    padding_width = (patch_size - image_size[1] % patch_size) % patch_size
    padding_height = (patch_size - image_size[0] % patch_size) % patch_size
    
    # Calculate feature map dimensions after padding
    feat_width = (image_size[1] + padding_width) // patch_size
    feat_height = (image_size[0] + padding_height) // patch_size
    
    if use_mask:
        return _assign_patches_from_masks(
            input_data['masks'],
            feat_height,
            feat_width,
            patch_size,
            padding_height,
            padding_width,
            coverage_threshold
        )
    else:
        return _assign_patches_from_boxes(
            input_data['boxes'],
            feat_height,
            feat_width,
            patch_size
        )

def _assign_patches_from_masks(masks, feat_height, feat_width, patch_size, 
                             padding_height, padding_width, coverage_threshold):
    """Helper function to assign patches based on mask coverage."""
    num_masks = len(masks)
    patch_assignments = {}
    
    # Track patch assignments and coverage ratios
    assigned_patches = np.full((feat_height, feat_width), -1)
    mask_coverage = np.zeros((num_masks, feat_height, feat_width))
    
    # Calculate coverage ratios for all masks
    for mask_idx, mask in enumerate(masks):
        # Convert tensor to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure mask is 2D
        mask = mask.squeeze() if mask.ndim > 2 else mask
        
        # Pad and reshape mask into patches
        padded_mask = np.pad(
            mask,
            ((0, padding_height), (0, padding_width)),
            mode='constant'
        )
        patch_grid = padded_mask.reshape(feat_height, patch_size, feat_width, patch_size)
        mask_coverage[mask_idx] = patch_grid.mean(axis=(1, 3))
    
    # Assign patches to masks with highest coverage
    for row in range(feat_height):
        for col in range(feat_width):
            coverage_ratios = mask_coverage[:, row, col]
            valid_masks = np.where(coverage_ratios > coverage_threshold)[0]
            
            if len(valid_masks) > 0:
                best_mask = valid_masks[np.argmax(coverage_ratios[valid_masks])]
                assigned_patches[row, col] = best_mask
    
    # Convert assignments to patch indices
    for row in range(feat_height):
        for col in range(feat_width):
            mask_idx = assigned_patches[row, col]
            if mask_idx != -1:
                patch_idx = row * feat_width + col
                if mask_idx in patch_assignments:
                    patch_assignments[mask_idx].append(patch_idx)
                else:
                    patch_assignments[mask_idx] = [patch_idx]
    
    # Ensure all masks have an entry
    for mask_idx in range(num_masks):
        if mask_idx not in patch_assignments:
            patch_assignments[mask_idx] = None
            
    return patch_assignments

def _assign_patches_from_boxes(boxes, feat_height, feat_width, patch_size):
    """Helper function to assign patches based on bounding boxes."""
    patch_assignments = {}
    
    for box_idx, box in enumerate(boxes):
        # Convert tensor to numpy if needed
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        # Calculate patch coordinates
        start_x = int(box[0] // patch_size)
        start_y = int(box[1] // patch_size)
        end_x = min(int(box[2] // patch_size), feat_width - 1)
        end_y = min(int(box[3] // patch_size), feat_height - 1)
        
        # Collect all patch indices within box
        indices = [
            y * feat_width + x 
            for y in range(start_y, end_y + 1)
            for x in range(start_x, end_x + 1)
        ]
        patch_assignments[box_idx] = indices
        
    return patch_assignments


def filter_background_masks(similarity_matrix, patch_assignments, ref_patch_id, 
                          mask_data, similarity_threshold=0.8):
    """
    Filter out background masks based on cosine similarity with a reference patch.
    
    Args:
        similarity_matrix: torch.Tensor [batch_size, num_patches, num_patches]
            Cosine similarity matrix between patches
        patch_assignments: dict
            Mapping of mask indices to their assigned patch indices (or None)
        ref_patch_id: int
            Index of the reference patch (assumed to be background)
        mask_data: dict
            Dictionary containing 'masks' and 'boxes' tensors
        similarity_threshold: float
            Threshold for considering a mask as background based on average similarity
    
    Returns:
        dict: Filtered mask data containing only non-background masks and boxes
    """
    background_masks = {}
    background_patches = {}
    foreground_patches = {}
    
    # Process each mask and its assigned patches
    for mask_idx, patch_indices in patch_assignments.items():
        # Handle masks with no assigned patches
        if patch_indices is None or not patch_indices:
            background_masks[mask_idx] = True
            continue
            
        # Calculate average similarity to reference patch
        patch_similarities = [
            similarity_matrix[0, ref_patch_id, idx] 
            for idx in patch_indices
        ]
        
        avg_similarity = sum(patch_similarities) / len(patch_similarities)
        is_background = avg_similarity > similarity_threshold
        
        # Store results
        background_masks[mask_idx] = is_background
        if is_background:
            background_patches[mask_idx] = patch_indices
        else:
            foreground_patches[mask_idx] = patch_indices
    
    # Filter masks and boxes
    foreground_indices = [
        idx for idx, is_background in background_masks.items() 
        if not is_background
    ]
    
    return {
        'masks': mask_data['masks'][foreground_indices],
        'boxes': mask_data['boxes'][foreground_indices]
    }

def filter_masks(
    mask_dict: dict,
    cosine_similarity: torch.Tensor,
    reference_patch_ids: torch.Tensor,
    image_size: tuple[int, int],
    vit_patch_size: int = 14,
    mean_threshold: float = 0.5,
    redundancy_threshold: float = 0.8,
    patch_coverage_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Filter and process image masks by applying size, redundancy, and background filtering.

    Args:
        mask_dict: Dictionary containing masks and their corresponding bounding boxes
        cosine_similarity: Tensor containing cosine similarity scores between patches
        reference_patch_ids: Tensor containing reference patch identifiers
        image_size: Tuple of (height, width) representing the input image dimensions
        patch_size: Size of each patch (default: 14)
        mean_threshold: Threshold for mean similarity score (default: 0.5)
        redundancy_threshold: Threshold for determining mask redundancy (default: 0.8)
        patch_coverage_threshold: Threshold for patch coverage in mask (default: 0.1)

    Returns:
        torch.Tensor: Filtered and processed masks
    """
    # Process the largest connected component in each mask
    processed_masks = extract_largest_component(mask_dict['masks'])
    bounding_boxes = calculate_bounding_boxes(masks=processed_masks)
    processed_masks = {'masks': processed_masks, 'boxes': bounding_boxes}

    # Remove redundant masks
    processed_masks = process_overlapping_masks(processed_masks, overlap_threshold=redundancy_threshold)

    # Get patch indices based on mask coverage
    mask_patches = calculate_patch_assignments(
                                                processed_masks,
                                                image_size=image_size,
                                                patch_size=vit_patch_size,
                                                use_mask=True,
                                                coverage_threshold=patch_coverage_threshold
                                                )

    # Filter out background masks based on similarity scores
    filtered_masks = filter_background_masks(
                                                similarity_matrix=cosine_similarity,
                                                patch_assignments=mask_patches,
                                                ref_patch_id=reference_patch_ids,
                                                mask_data=processed_masks,
                                                similarity_threshold=mean_threshold
                                            )['masks']

    return filtered_masks