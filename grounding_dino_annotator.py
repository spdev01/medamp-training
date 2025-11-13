"""
Grounding DINO Annotator Module
Handles object detection using Grounding DINO model.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundingDINOAnnotator:
    """
    Use Grounding DINO for zero-shot object detection.
    Generates bounding boxes for specified text prompts.
    """
    
    def __init__(
        self,
        model_config_path: Optional[str] = None,
        model_checkpoint_path: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Grounding DINO model.
        
        Args:
            model_config_path: Path to model config (uses default if None)
            model_checkpoint_path: Path to model checkpoint (uses default if None)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model = None
        
        logger.info(f"Initializing Grounding DINO on {device}")
        self._load_model(model_config_path, model_checkpoint_path)
    
    def _load_model(self, config_path: Optional[str], checkpoint_path: Optional[str]):
        """Load the Grounding DINO model."""
        try:
            from groundingdino.util.inference import load_model
            from pathlib import Path
            
            # Use default paths if not provided
            if config_path is None:
                config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            if checkpoint_path is None:
                checkpoint_path = "weights/groundingdino_swint_ogc.pth"
            
            # Verify paths exist
            config_path = Path(config_path)
            checkpoint_path = Path(checkpoint_path)
            
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                logger.error(f"Please check the path in config.yaml")
                self.model = None
                return
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                logger.error(f"Please download the weights to: {checkpoint_path}")
                self.model = None
                return
            
            logger.info(f"Loading model from: {checkpoint_path}")
            logger.info(f"Using config: {config_path}")
            
            self.model = load_model(str(config_path), str(checkpoint_path), device=self.device)
            logger.info("Grounding DINO model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Grounding DINO import error: {str(e)}")
            logger.warning(
                "Grounding DINO not installed properly. "
                "Make sure it's installed in your Python environment"
            )
            self.model = None
        except Exception as e:
            logger.error(f"Error loading Grounding DINO model: {str(e)}")
            logger.error(f"Config path: {config_path}")
            logger.error(f"Checkpoint path: {checkpoint_path}")
            self.model = None
    
    def annotate_image(
        self,
        image_path: str,
        text_prompt: str,
        return_labels: bool = True
    ) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Detect objects in an image using text prompt.
        
        Args:
            image_path: Path to the image file
            text_prompt: Text description of objects to detect (e.g., "person . car . dog")
            return_labels: Whether to return detected class labels
            
        Returns:
            Tuple of (boxes, labels, scores)
            - boxes: List of [x_min, y_min, x_max, y_max] in absolute coordinates
            - labels: List of detected class names
            - scores: List of confidence scores
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot perform detection.")
            return [], [], []
        
        try:
            from groundingdino.util.inference import predict, load_image
            
            # Load and transform image
            image_source, image_transformed = load_image(image_path)
            
            # Perform detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # Convert boxes from normalized to absolute coordinates
            h, w, _ = image_source.shape
            boxes_abs = []
            for box in boxes:
                # box is in format [cx, cy, w, h] normalized
                cx, cy, bw, bh = box
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                x_max = (cx + bw / 2) * w
                y_max = (cy + bh / 2) * h
                boxes_abs.append([x_min, y_min, x_max, y_max])
            
            labels = [phrase.strip() for phrase in phrases]
            scores = logits.tolist()
            
            # Keep only the detection with highest confidence
            if len(boxes_abs) > 0:
                max_idx = scores.index(max(scores))
                boxes_abs = [boxes_abs[max_idx]]
                labels = [labels[max_idx]]
                scores = [scores[max_idx]]
                logger.info(f"Detected {len(boxes_abs)} object in {Path(image_path).name} (kept highest confidence: {scores[0]:.3f})")
            else:
                logger.info(f"No objects detected in {Path(image_path).name}")
            
            return boxes_abs, labels, scores
            
        except Exception as e:
            logger.error(f"Error annotating {image_path}: {str(e)}")
            return [], [], []
    
    def annotate_batch(
        self,
        image_paths: List[str],
        text_prompt: str,
        show_progress: bool = True
    ) -> Dict[str, Tuple[List[List[float]], List[str], List[float]]]:
        """
        Annotate multiple images.
        
        Args:
            image_paths: List of image file paths
            text_prompt: Text description of objects to detect
            show_progress: Whether to show progress logs
            
        Returns:
            Dictionary mapping image paths to (boxes, labels, scores) tuples
        """
        results = {}
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths, 1):
            if show_progress and idx % 10 == 0:
                logger.info(f"Processing image {idx}/{total}")
            
            boxes, labels, scores = self.annotate_image(image_path, text_prompt)
            results[image_path] = (boxes, labels, scores)
        
        return results
    
    @staticmethod
    def get_default_prompts() -> Dict[str, str]:
        """
        Get common detection prompts for different use cases.
        
        Returns:
            Dictionary of prompt names to prompt strings
        """
        return {
            "general": "person . car . bicycle . motorcycle . bus . truck . traffic light . stop sign",
            "medical": "lesion . tumor . abnormality . nodule . mass . cyst",
            "people": "person . face . hand . body",
            "vehicles": "car . truck . bus . motorcycle . bicycle . boat . airplane",
            "animals": "dog . cat . bird . horse . cow . sheep . elephant",
            "custom": "object"  # User should replace with their own
        }
