"""
Video Frame Extraction and Auto-Annotation Pipeline
Main script for extracting frames from videos and generating YOLO-format annotations using Grounding DINO.
"""
import argparse
import yaml
from pathlib import Path
import logging
from typing import Optional
from PIL import Image

from frame_extractor import VideoFrameExtractor
from grounding_dino_annotator import GroundingDINOAnnotator
from yolo_formatter import YOLOFormatter
from visualizer import BoundingBoxVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPrepPipeline:
    """
    Complete pipeline for preparing video data for YOLO training.
    """
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.frame_extractor = VideoFrameExtractor(
            output_dir=config['paths']['frames_output']
        )
        
        self.annotator = GroundingDINOAnnotator(
            model_config_path=config['grounding_dino'].get('config_path'),
            model_checkpoint_path=config['grounding_dino'].get('checkpoint_path'),
            box_threshold=config['grounding_dino']['box_threshold'],
            text_threshold=config['grounding_dino']['text_threshold']
        )
        
        self.yolo_formatter = YOLOFormatter(
            output_dir=config['paths']['annotations_output']
        )
        
        self.visualizer = BoundingBoxVisualizer(
            output_dir=config['paths']['visualizations_output']
        )
    
    def run(self):
        """Execute the complete pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Video Preparation Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Get frame paths (skip extraction if already exists)
        frames_dir = Path(self.config['paths']['frames_output'])
        existing_frames = list(frames_dir.glob('**/*.jpg'))
        
        if existing_frames:
            logger.info(f"\n[STEP 1/4] Using existing {len(existing_frames)} frames from {frames_dir}")
            all_frame_paths = existing_frames
        else:
            logger.info("\n[STEP 1/4] Extracting frames from videos...")
            frame_results = self.frame_extractor.extract_from_directory(
                video_dir=self.config['paths']['video_input'],
                frame_interval=self.config['extraction']['frame_interval'],
                max_frames=self.config['extraction'].get('max_frames_per_video'),
                target_size=self.config['extraction'].get('target_size')
            )
            
            # Flatten frame paths
            all_frame_paths = []
            for video_name, paths in frame_results.items():
                all_frame_paths.extend(paths)
        
        logger.info(f"Extracted {len(all_frame_paths)} frames total")
        
        if len(all_frame_paths) == 0:
            logger.error("No frames extracted. Exiting.")
            return
        
        # Step 2 & 3: Annotate with Grounding DINO and save immediately
        logger.info("\n[STEP 2/4] Annotating frames with Grounding DINO and saving annotations...")
        text_prompt = self.config['grounding_dino']['text_prompt']
        logger.info(f"Detection prompt: '{text_prompt}'")
        
        annotation_count = 0
        detection_count = 0
        total = len(all_frame_paths)
        detection_results = {}  # Store for visualization later
        
        for idx, frame_path in enumerate(all_frame_paths, 1):
            if idx % 10 == 0:
                logger.info(f"Processing image {idx}/{total} ({detection_count} detections so far)")
            
            # Annotate single image
            boxes, labels, scores = self.annotator.annotate_image(str(frame_path), text_prompt)
            
            # Store for visualization
            detection_results[str(frame_path)] = (boxes, labels, scores)
            
            # Save immediately if detections found
            if len(boxes) > 0:
                detection_count += len(boxes)
                
                # Get image dimensions
                img = Image.open(frame_path)
                width, height = img.size
                
                # Save YOLO annotation immediately
                self.yolo_formatter.save_annotation(
                    image_path=str(frame_path),
                    boxes=boxes,
                    labels=labels,
                    image_width=width,
                    image_height=height,
                    scores=scores,
                    min_confidence=self.config['grounding_dino'].get('min_confidence', 0.0)
                )
                annotation_count += 1
                
                # Create visualization immediately if enabled
                if self.config['visualization']['enabled']:
                    try:
                        self.visualizer.draw_boxes(
                            image_path=str(frame_path),
                            boxes=boxes,
                            labels=labels,
                            scores=scores,
                            show_confidence=self.config['visualization']['show_confidence']
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create visualization for {frame_path}: {e}")
        
        logger.info(f"\nProcessed {total} images")
        logger.info(f"Created {annotation_count} annotation files with {detection_count} total detections")
        
        # Save class names
        class_names_path = self.yolo_formatter.save_class_names()
        logger.info(f"Class names saved to: {class_names_path}")
        
        # Create YAML config
        yaml_path = self.yolo_formatter.create_yaml_config(
            train_images_path=self.config['paths']['frames_output']
        )
        logger.info(f"YOLO config saved to: {yaml_path}")
        
        # Print class statistics
        logger.info("\nClass distribution:")
        for class_name, class_id in self.yolo_formatter.get_class_statistics().items():
            logger.info(f"  {class_id}: {class_name}")
        
        # Step 4: Create grid visualization
        logger.info("\n[STEP 4/4] Creating grid visualization...")
        
        if self.config['visualization']['enabled'] and self.config['visualization'].get('create_grid', True):
            # Filter to only images with detections
            valid_samples = [
                (p, boxes, labels, scores) 
                for p, (boxes, labels, scores) in detection_results.items()
                if len(boxes) > 0
            ]
            
            if valid_samples:
                # Take up to 9 samples
                valid_samples = valid_samples[:9]
                paths, boxes_list, labels_list, scores_list = zip(*valid_samples)
                
                self.visualizer.create_grid_visualization(
                    image_paths=list(paths),
                    all_boxes=list(boxes_list),
                    all_labels=list(labels_list),
                    all_scores=list(scores_list)
                )
                logger.info(f"Created grid visualization with {len(valid_samples)} images")
            else:
                logger.warning("No detections found to create grid visualization")
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nOutput locations:")
        logger.info(f"  Frames: {self.config['paths']['frames_output']}")
        logger.info(f"  Annotations: {self.config['paths']['annotations_output']}")
        logger.info(f"  Visualizations: {self.config['paths']['visualizations_output']}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review visualizations in: {self.config['paths']['visualizations_output']}")
        logger.info(f"  2. Check YOLO config: {yaml_path}")
        logger.info(f"  3. Start YOLO training with the generated dataset")


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and auto-annotate for YOLO training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directory containing video files (overrides config)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Grounding DINO detection prompt (overrides config)'
    )
    parser.add_argument(
        '--frame-interval',
        type=int,
        help='Extract every Nth frame (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.error(f"Config file not found: {args.config}")
        logger.info("Creating default config file...")
        create_default_config(args.config)
        logger.info(f"Default config created at {args.config}")
        logger.info("Please edit the config file and run again.")
        return
    
    # Override with command-line arguments
    if args.video_dir:
        config['paths']['video_input'] = args.video_dir
    if args.prompt:
        config['grounding_dino']['text_prompt'] = args.prompt
    if args.frame_interval:
        config['extraction']['frame_interval'] = args.frame_interval
    
    # Run pipeline
    pipeline = VideoPrepPipeline(config)
    pipeline.run()


def create_default_config(output_path: str):
    """Create a default configuration file."""
    default_config = {
        'paths': {
            'video_input': 'raw_footages',
            'frames_output': 'extracted_frames',
            'annotations_output': 'yolo_annotations',
            'visualizations_output': 'visualizations'
        },
        'extraction': {
            'frame_interval': 30,  # Extract every 30th frame
            'max_frames_per_video': None,  # None = extract all
            'target_size': None  # None = keep original size, or [width, height]
        },
        'grounding_dino': {
            'config_path': None,  # None = use default
            'checkpoint_path': None,  # None = use default
            'text_prompt': 'person . car . object',  # Customize for your use case
            'box_threshold': 0.35,
            'text_threshold': 0.25,
            'min_confidence': 0.3  # Minimum confidence to save annotation
        },
        'visualization': {
            'enabled': True,
            'show_confidence': True,
            'max_images': 50,  # Maximum number of images to visualize
            'create_grid': True
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    main()
