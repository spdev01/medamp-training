"""
Visualization Module
Draw bounding boxes on images for verification.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundingBoxVisualizer:
    """
    Visualize bounding boxes on images for quality checking.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for different classes
        self.color_palette = self._generate_color_palette(100)
        self.class_colors = {}
    
    def _generate_color_palette(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate a palette of visually distinct colors.
        
        Args:
            n_colors: Number of colors to generate
            
        Returns:
            List of RGB color tuples
        """
        colors = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_colors):
            # Use HSV for better color distribution
            hue = (i * 137.508) % 360  # Golden angle
            saturation = 0.7 + (np.random.rand() * 0.3)
            value = 0.7 + (np.random.rand() * 0.3)
            
            # Convert HSV to RGB
            h = hue / 60.0
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c
            
            if 0 <= h < 1:
                r, g, b = c, x, 0
            elif 1 <= h < 2:
                r, g, b = x, c, 0
            elif 2 <= h < 3:
                r, g, b = 0, c, x
            elif 3 <= h < 4:
                r, g, b = 0, x, c
            elif 4 <= h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
            colors.append((r, g, b))
        
        return colors
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get a consistent color for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            RGB color tuple
        """
        if class_name not in self.class_colors:
            color_idx = len(self.class_colors) % len(self.color_palette)
            self.class_colors[class_name] = self.color_palette[color_idx]
        
        return self.class_colors[class_name]
    
    def draw_boxes(
        self,
        image_path: str,
        boxes: List[List[float]],
        labels: List[str],
        scores: Optional[List[float]] = None,
        thickness: int = 2,
        font_scale: float = 0.5,
        show_confidence: bool = True
    ) -> str:
        """
        Draw bounding boxes on an image and save the result.
        
        Args:
            image_path: Path to the input image
            boxes: List of boxes in [x_min, y_min, x_max, y_max] format
            labels: List of class labels for each box
            scores: List of confidence scores (optional)
            thickness: Line thickness for bounding boxes
            font_scale: Font scale for text labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Path to the saved visualization image
        """
        image_path = Path(image_path)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None
        
        # Draw each bounding box
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Get color for this class
            color = self.get_class_color(label)
            
            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # Prepare label text
            if scores and show_confidence:
                text = f"{label}: {scores[idx]:.2f}"
            else:
                text = label
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=1
            )
            
            # Draw background rectangle for text
            text_y = y_min - 10 if y_min > 20 else y_min + text_height + 10
            cv2.rectangle(
                image,
                (x_min, text_y - text_height - 5),
                (x_min + text_width + 5, text_y + 5),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                text,
                (x_min + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Add summary text at the top
        summary_text = f"Detected: {len(boxes)} objects"
        cv2.putText(
            image,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Save visualization
        vis_filename = f"{image_path.stem}_annotated.jpg"
        vis_path = self.output_dir / vis_filename
        cv2.imwrite(str(vis_path), image)
        
        logger.debug(f"Saved visualization to {vis_filename}")
        
        return str(vis_path)
    
    def create_grid_visualization(
        self,
        image_paths: List[str],
        all_boxes: List[List[List[float]]],
        all_labels: List[List[str]],
        all_scores: Optional[List[List[float]]] = None,
        grid_size: Tuple[int, int] = (3, 3),
        output_filename: str = "grid_visualization.jpg"
    ) -> str:
        """
        Create a grid visualization of multiple annotated images.
        
        Args:
            image_paths: List of image paths
            all_boxes: List of box lists for each image
            all_labels: List of label lists for each image
            all_scores: List of score lists for each image (optional)
            grid_size: (rows, cols) for the grid
            output_filename: Name of the output file
            
        Returns:
            Path to the saved grid visualization
        """
        rows, cols = grid_size
        max_images = rows * cols
        
        # Limit to grid size
        image_paths = image_paths[:max_images]
        all_boxes = all_boxes[:max_images]
        all_labels = all_labels[:max_images]
        if all_scores:
            all_scores = all_scores[:max_images]
        
        # Read and annotate images
        annotated_images = []
        for idx, (img_path, boxes, labels) in enumerate(zip(image_paths, all_boxes, all_labels)):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            scores = all_scores[idx] if all_scores else None
            
            # Draw boxes on this image
            for box_idx, (box, label) in enumerate(zip(boxes, labels)):
                x_min, y_min, x_max, y_max = map(int, box)
                color = self.get_class_color(label)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Add label
                if scores:
                    text = f"{label}: {scores[box_idx]:.2f}"
                else:
                    text = label
                
                cv2.putText(
                    image, text, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
                )
            
            annotated_images.append(image)
        
        if not annotated_images:
            logger.error("No valid images to create grid")
            return None
        
        # Resize all images to same size
        target_size = (640, 480)
        resized_images = [cv2.resize(img, target_size) for img in annotated_images]
        
        # Pad with blank images if needed
        while len(resized_images) < max_images:
            blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            resized_images.append(blank)
        
        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = resized_images[i * cols:(i + 1) * cols]
            row = np.hstack(row_images)
            grid_rows.append(row)
        
        grid = np.vstack(grid_rows)
        
        # Save grid
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), grid)
        
        logger.info(f"Created grid visualization at {output_path}")
        return str(output_path)
