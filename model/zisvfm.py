import torch
import numpy as np
import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from .segmentor_model import CustomSamAutomaticMaskGenerator
from .segmentor_model import load_sam
from .descriptor_model import DescriptorModel
from .filter_masks import filter_masks

class Zisvfm:
    """
    ZISVFM: Zero-Shot Object Instance Segmentation in Indoor Robotic Environments with Vision Foundation Models
    Handles image segmentation using SAM (Segment Anything Model) and descriptor-based filtering.
    """
    
    def __init__(self, config_path: str = '/path/to/configs.yaml'):
        """
        Initialize the UOIS model with configurations.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and print configurations
        with open(config_path, 'r') as file:
            self.configs = yaml.safe_load(file)
        
        print("Loading configurations...")
        for key, value in sorted(self.configs.items()):
            print(f"{key}: {value}")

        # Initialize models
        self._initialize_models()
        self._move_models_to_device()
        print(f"Models successfully moved to {self.device}")

    def _initialize_models(self):
        """Initialize SAM and descriptor models with configuration parameters."""
        # Load SAM model
        self.sam = load_sam(
            self.configs["SegmentorModel"]["paths"]["model_type"],
            self.configs["SegmentorModel"]["paths"]["weights_dir"]
        )
        
        # Initialize segmentor model with SAM
        self.segmentor_model = CustomSamAutomaticMaskGenerator(
            sam=self.sam,
            **self.configs["SegmentorModel"]["parameters"]
        )
        
        # Initialize descriptor model
        self.descriptor_model = DescriptorModel(**self.configs["DescriptorModel"])
        
        # Store configuration parameters
        self.filter_masks_config = self.configs["FilterMasks"]

    def _move_models_to_device(self):
        """Move all models to the specified device (CPU/GPU)."""
        self.descriptor_model.vit_encoder.model = self.descriptor_model.vit_encoder.model.to(self.device)
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = self.segmentor_model.predictor.model.to(self.device)
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)

    def forward_step(self, depth_path: str, rgb_path: str):
        """
        Process an image pair (RGB and depth) through the segmentation pipeline.
        
        Args:
            depth_path (str): Path to the depth image
            rgb_path (str): Path to the RGB image
            return_debug (bool): Whether to return additional debug information
            
        Returns:
            torch.Tensor: Segmentation masks
            dict (optional): Debug information if return_debug is True
        """
        # Convert depth to colored image and preprocess RGB
        depth_colored = self._depth_to_colored_image(depth_path)
        preprocessed_rgb = self._preprocess_image(rgb_path)

        # Extract features using descriptor model
        input_tensor, shape_features = self.descriptor_model.preprocess(preprocessed_rgb)
        attention, key = self.descriptor_model.extract_feats(input_tensor, ["attn", "key"])
        id_patch_ref, cos_sim_weighted = self.descriptor_model.get_background_patch_index(attention, key, shape_features)

        # Generate and filter proposals
        proposals = self.segmentor_model.generate_masks(depth_colored)
        proposals = self._filter_noise_masks(proposals, depth_path, depth_threshold=200)
        
        proposals = filter_masks(
            proposals,
            cos_sim_weighted,
            id_patch_ref,
            **self.filter_masks_config
        )

        if proposals.size(0) > 0:
            return proposals
        else:
            empty_mask = torch.zeros(1, 480, 640, dtype=torch.bool, device=self.device)
            return empty_mask

    def _preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image_input: Path to image or PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        
        Raises:
            TypeError: If input is neither a string path nor a PIL Image
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError("Input must be a file path or a PIL image")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        preprocessed = torch.utils.data.default_collate([transform(image)])
        return preprocessed.to(self.device)

    def _depth_to_colored_image(self, depth_path: str, colormap: str = 'inferno') -> np.ndarray:
        """
        Convert a depth image to a colored representation.
        
        Args:
            depth_path (str): Path to depth image
            colormap (str): Matplotlib colormap name
            
        Returns:
            np.ndarray: Colored depth image (RGB format)
        """
        depth_img = cv2.imread(depth_path, -1)
        depth_mask = (depth_img != 0)
        
        # Normalize depth values
        normalized = np.zeros_like(depth_img, dtype=float)
        normalized[depth_mask] = (1 / (depth_img + 1e-6))[depth_mask]
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        # Apply colormap
        colored = (plt.get_cmap(colormap)(normalized) * 255).astype(np.uint8)
        return colored[:, :, :3]

    def _filter_noise_masks(self, proposals: dict, depth_path: str, depth_threshold: float = 100) -> dict:
        """
        Filter out noise masks based on depth information.
        
        Args:
            proposals (dict): Dictionary containing 'masks' and 'boxes'
            depth_path (str): Path to depth image
            depth_threshold (float): Minimum depth threshold for valid masks
            
        Returns:
            dict: Filtered proposals
        """
        masks = proposals["masks"]
        boxes = proposals["boxes"]
        
        # Load and process depth image
        depth_image = torch.from_numpy(np.array(Image.open(depth_path))).to(self.device).float()
        
        # Calculate depth statistics for each mask
        mask_depths = []
        for mask in masks:
            masked_depth = depth_image[mask.bool()]
            if masked_depth.numel() > 0:
                sorted_depth = torch.sort(masked_depth)[0]
                quarter_idx = int(masked_depth.numel() * 0.25)
                quarter_mean = sorted_depth[:quarter_idx].mean().item()
                mask_depths.append(quarter_mean)
            else:
                mask_depths.append(0)
        
        # Filter masks based on depth threshold
        mask_depths = torch.tensor(mask_depths, device=self.device)
        valid_indices = mask_depths >= depth_threshold
        
        return {
            "masks": masks[valid_indices],
            "boxes": boxes[valid_indices]
        }