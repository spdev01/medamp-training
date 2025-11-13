"""
YOLO Format Converter Module
Converts bounding box annotations to YOLO format for training.
"""
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOFormatter:
    """
    Convert bounding box detections to YOLO format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1).
    """
    
    def __init__(self, output_dir: str = "yolo_annotations", class_names: List[str] = None):
        """
        Initialize YOLO formatter.
        
        Args:
            output_dir: Directory to save YOLO annotation files
            class_names: List of class names (order determines class IDs)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or []
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
    
    def add_class(self, class_name: str) -> int:
        """
        Add a new class to the class list.
        
        Args:
            class_name: Name of the class to add
            
        Returns:
            Class ID assigned to this class
        """
        if class_name not in self.class_to_id:
            class_id = len(self.class_names)
            self.class_names.append(class_name)
            self.class_to_id[class_name] = class_id
            return class_id
        return self.class_to_id[class_name]
    
    def convert_box_to_yolo(
        self,
        box: List[float],
        image_width: int,
        image_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert absolute box coordinates to YOLO normalized format.
        
        Args:
            box: [x_min, y_min, x_max, y_max] in absolute coordinates
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized to 0-1
        """
        x_min, y_min, x_max, y_max = box
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize by image dimensions
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        # Clamp to [0, 1]
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def save_annotation(
        self,
        image_path: str,
        boxes: List[List[float]],
        labels: List[str],
        image_width: int,
        image_height: int,
        min_confidence: float = 0.0,
        scores: List[float] = None
    ) -> str:
        """
        Save annotations for an image in YOLO format.
        
        Args:
            image_path: Path to the image file
            boxes: List of boxes in [x_min, y_min, x_max, y_max] format
            labels: List of class labels for each box
            image_width: Width of the image
            image_height: Height of the image
            min_confidence: Minimum confidence score to include a detection
            scores: List of confidence scores (optional)
            
        Returns:
            Path to the saved annotation file
        """
        image_path = Path(image_path)
        annotation_filename = image_path.stem + ".txt"
        annotation_path = self.output_dir / annotation_filename
        
        # Extract class ID from image filename (number before underscore)
        # Example: "1_30.jpg" -> class_id = 1
        filename_stem = image_path.stem
        try:
            class_id = int(filename_stem.split('_')[0])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract class ID from filename {filename_stem}, using 0")
            class_id = 0
        
        lines = []
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            # Skip if confidence is too low
            if scores and scores[idx] < min_confidence:
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = self.convert_box_to_yolo(
                box, image_width, image_height
            )
            
            # YOLO format line
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)
        
        # Write to file
        with open(annotation_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.debug(f"Saved {len(lines)} annotations to {annotation_filename}")
        
        return str(annotation_path)
    
    def save_class_names(self, output_path: str = None) -> str:
        """
        Save class names to a file (one per line).
        This file is needed for YOLO training.
        
        Args:
            output_path: Path to save class names (default: classes.txt in output_dir)
            
        Returns:
            Path to the saved class names file
        """
        if output_path is None:
            output_path = self.output_dir / "classes.txt"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        logger.info(f"Saved {len(self.class_names)} class names to {output_path}")
        return str(output_path)
    
    def create_yaml_config(
        self,
        train_images_path: str,
        val_images_path: str = None,
        output_path: str = None
    ) -> str:
        """
        Create a YAML config file for YOLO training.
        
        Args:
            train_images_path: Path to training images directory
            val_images_path: Path to validation images directory (optional)
            output_path: Path to save YAML file (default: dataset.yaml in output_dir)
            
        Returns:
            Path to the saved YAML config file
        """
        if output_path is None:
            output_path = self.output_dir / "dataset.yaml"
        else:
            output_path = Path(output_path)
        
        # Prepare YAML content
        yaml_content = f"""# YOLO Dataset Configuration
path: {Path(train_images_path).parent.absolute()}  # dataset root dir
train: {Path(train_images_path).name}  # train images (relative to 'path')
"""
        
        if val_images_path:
            yaml_content += f"val: {Path(val_images_path).name}  # val images (relative to 'path')\n"
        
        yaml_content += f"\n# Classes\nnames:\n"
        for idx, name in enumerate(self.class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        # Write YAML file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created YOLO config at {output_path}")
        return str(output_path)
    
    def get_class_statistics(self) -> Dict[str, int]:
        """
        Get statistics about annotated classes.
        
        Returns:
            Dictionary mapping class names to their IDs
        """
        return self.class_to_id.copy()
