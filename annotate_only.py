"""
Annotation-Only Script
Run Grounding DINO annotation on existing frames without extracting new ones.
"""
import yaml
from pathlib import Path
import logging
from PIL import Image

from grounding_dino_annotator import GroundingDINOAnnotator
from yolo_formatter import YOLOFormatter
from visualizer import BoundingBoxVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("Starting Annotation Pipeline (Frames Only)")
    logger.info("=" * 80)
    
    # Get all existing frames
    frames_dir = Path(config['paths']['frames_output'])
    all_frames = list(frames_dir.rglob('*.jpg'))
    logger.info(f"Found {len(all_frames)} frames to annotate")
    
    if len(all_frames) == 0:
        logger.error(f"No frames found in {frames_dir}")
        return
    
    # Initialize components
    logger.info("\n[STEP 1/4] Initializing Grounding DINO...")
    annotator = GroundingDINOAnnotator(
        model_config_path=config['grounding_dino'].get('config_path'),
        model_checkpoint_path=config['grounding_dino'].get('checkpoint_path'),
        box_threshold=config['grounding_dino']['box_threshold'],
        text_threshold=config['grounding_dino']['text_threshold']
    )
    
    yolo_formatter = YOLOFormatter(
        output_dir=config['paths']['annotations_output']
    )
    
    visualizer = BoundingBoxVisualizer(
        output_dir=config['paths']['visualizations_output']
    )
    
    # Annotate with Grounding DINO
    logger.info("\n[STEP 2/4] Annotating frames with Grounding DINO...")
    text_prompt = config['grounding_dino']['text_prompt']
    logger.info(f"Detection prompt: '{text_prompt}'")
    
    detection_results = annotator.annotate_batch(
        image_paths=[str(p) for p in all_frames],
        text_prompt=text_prompt,
        show_progress=True
    )
    
    # Convert to YOLO format
    logger.info("\n[STEP 3/4] Converting annotations to YOLO format...")
    annotation_count = 0
    
    for image_path, (boxes, labels, scores) in detection_results.items():
        if len(boxes) == 0:
            continue
        
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Save YOLO annotation
        yolo_formatter.save_annotation(
            image_path=image_path,
            boxes=boxes,
            labels=labels,
            image_width=width,
            image_height=height,
            scores=scores,
            min_confidence=config['grounding_dino'].get('min_confidence', 0.0)
        )
        annotation_count += 1
    
    logger.info(f"Created {annotation_count} annotation files")
    
    # Save class names
    class_names_path = yolo_formatter.save_class_names()
    logger.info(f"Class names saved to: {class_names_path}")
    
    # Create YAML config
    yaml_path = yolo_formatter.create_yaml_config(
        train_images_path=config['paths']['frames_output']
    )
    logger.info(f"YOLO config saved to: {yaml_path}")
    
    # Print class statistics
    logger.info("\nClass distribution:")
    for class_name, class_id in yolo_formatter.get_class_statistics().items():
        logger.info(f"  {class_id}: {class_name}")
    
    # Create visualizations
    logger.info("\n[STEP 4/4] Creating visualizations...")
    
    if config['visualization']['enabled']:
        vis_count = 0
        max_vis = config['visualization'].get('max_images', 50)
        
        for image_path, (boxes, labels, scores) in list(detection_results.items())[:max_vis]:
            if len(boxes) > 0:
                visualizer.draw_boxes(
                    image_path=image_path,
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    show_confidence=config['visualization']['show_confidence']
                )
                vis_count += 1
        
        logger.info(f"Created {vis_count} visualization images")
        
        # Create grid visualization
        if config['visualization'].get('create_grid', True):
            sample_paths = list(detection_results.keys())[:9]
            sample_boxes = [detection_results[p][0] for p in sample_paths]
            sample_labels = [detection_results[p][1] for p in sample_paths]
            sample_scores = [detection_results[p][2] for p in sample_paths]
            
            # Filter to only images with detections
            valid_samples = [
                (p, b, l, s) for p, b, l, s in zip(sample_paths, sample_boxes, sample_labels, sample_scores)
                if len(b) > 0
            ]
            
            if valid_samples:
                paths, boxes, labels, scores = zip(*valid_samples)
                visualizer.create_grid_visualization(
                    image_paths=list(paths),
                    all_boxes=list(boxes),
                    all_labels=list(labels),
                    all_scores=list(scores)
                )
    
    logger.info("\n" + "=" * 80)
    logger.info("Annotation completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nOutput locations:")
    logger.info(f"  Frames: {config['paths']['frames_output']}")
    logger.info(f"  Annotations: {config['paths']['annotations_output']}")
    logger.info(f"  Visualizations: {config['paths']['visualizations_output']}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review visualizations in: {config['paths']['visualizations_output']}")
    logger.info(f"  2. Check YOLO config: {yaml_path}")
    logger.info(f"  3. Start YOLO training with the generated dataset")


if __name__ == '__main__':
    main()
