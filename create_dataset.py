"""
Dataset Creation Script for YOLO Training
Creates train/test/valid split with class balancing for the training set.
"""
import argparse
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODatasetBuilder:
    """
    Build a YOLO dataset with train/test/valid splits.
    Ensures class balance in the training set.
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        output_dir: str = "dataset",
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        balance_classes: bool = True,
        seed: int = 42
    ):
        """
        Initialize dataset builder.
        
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing YOLO annotation files
            output_dir: Output directory for dataset
            train_ratio: Ratio of data for training (default: 0.8)
            valid_ratio: Ratio of data for validation (default: 0.1)
            test_ratio: Ratio of data for testing (default: 0.1)
            balance_classes: Whether to balance class distribution by sampling
            seed: Random seed for reproducibility
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.should_balance_classes = balance_classes
        
        # Validate ratios
        total = train_ratio + test_ratio + valid_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        random.seed(seed)
        
        # Create output structure
        self.splits = ['train', 'test', 'valid']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def get_class_from_annotation(self, annotation_path: Path) -> int:
        """
        Extract class ID from annotation file.
        Assumes first number in first line is the class ID.
        """
        try:
            with open(annotation_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    return int(first_line.split()[0])
        except Exception as e:
            logger.warning(f"Could not read class from {annotation_path}: {e}")
        return -1
    
    def group_by_class(self, image_files: list) -> dict:
        """
        Group image files by their class ID.
        
        Returns:
            Dictionary mapping class_id to list of (image_path, annotation_path) tuples
        """
        class_groups = defaultdict(list)
        
        for image_path in image_files:
            # Find corresponding annotation
            annotation_path = self.annotations_dir / (image_path.stem + '.txt')
            
            if not annotation_path.exists():
                logger.warning(f"No annotation found for {image_path.name}, skipping")
                continue
            
            # Get class ID
            class_id = self.get_class_from_annotation(annotation_path)
            
            if class_id >= 0:
                class_groups[class_id].append((image_path, annotation_path))
        
        return class_groups
    
    def balance_classes(self, class_groups: dict) -> dict:
        """
        Balance classes by sampling to match the class with minimum samples.
        
        Args:
            class_groups: Dictionary of class_id -> list of (image, annotation) tuples
            
        Returns:
            Balanced dictionary with equal samples per class
        """
        if not self.should_balance_classes:
            return class_groups
        
        # Find minimum class size
        min_samples = min(len(samples) for samples in class_groups.values())
        
        logger.info(f"\nBalancing classes to {min_samples} samples each...")
        
        balanced_groups = {}
        for class_id, samples in class_groups.items():
            original_count = len(samples)
            # Randomly sample to match minimum
            random.shuffle(samples)
            balanced_groups[class_id] = samples[:min_samples]
            
            logger.info(
                f"  Class {class_id}: {original_count} -> {min_samples} "
                f"(removed {original_count - min_samples})"
            )
        
        return balanced_groups
    
    def create_splits(self, class_groups: dict, split_ratios: dict) -> dict:
        """
        Create train/test/valid splits with equal class distribution in each split.
        
        Args:
            class_groups: Dictionary of class_id -> list of (image, annotation) tuples
            split_ratios: Dictionary of split_name -> ratio
            
        Returns:
            Dictionary of split_name -> list of (image, annotation) tuples
        """
        splits = {name: [] for name in split_ratios.keys()}
        
        # For each class, split proportionally
        for class_id, samples in class_groups.items():
            random.shuffle(samples)
            n_samples = len(samples)
            
            # Calculate split sizes
            n_train = int(n_samples * split_ratios['train'])
            n_test = int(n_samples * split_ratios['test'])
            n_valid = n_samples - n_train - n_test
            
            # Split samples
            splits['train'].extend(samples[:n_train])
            splits['test'].extend(samples[n_train:n_train + n_test])
            splits['valid'].extend(samples[n_train + n_test:])
            
            logger.info(
                f"  Class {class_id}: {n_samples} total -> "
                f"train={n_train}, test={n_test}, valid={n_valid}"
            )
        
        # Shuffle each split
        for split_name in splits:
            random.shuffle(splits[split_name])
        
        return splits
    
    def copy_files(self, splits: dict):
        """
        Copy image and annotation files to their respective split directories.
        """
        for split_name, samples in splits.items():
            logger.info(f"Copying {len(samples)} files to {split_name} split...")
            
            for image_path, annotation_path in samples:
                # Copy image
                dest_image = self.output_dir / split_name / 'images' / image_path.name
                shutil.copy2(image_path, dest_image)
                
                # Copy annotation
                dest_annotation = self.output_dir / split_name / 'labels' / annotation_path.name
                shutil.copy2(annotation_path, dest_annotation)
    
    def create_data_yaml(self, class_groups: dict):
        """
        Create data.yaml config file for YOLO training.
        """
        # Get class names (use class ID as name if we don't have a mapping)
        class_ids = sorted(class_groups.keys())
        class_names = {class_id: f"class_{class_id}" for class_id in class_ids}
        
        # Create YAML content
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': class_names
        }
        
        # Write YAML file
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path
    
    def build(self):
        """
        Build the complete dataset with balanced splits.
        """
        logger.info("=" * 80)
        logger.info("Building YOLO Dataset")
        logger.info("=" * 80)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")
        
        if len(image_files) == 0:
            logger.error("No images found!")
            return
        
        # Group by class
        logger.info("Grouping images by class...")
        class_groups = self.group_by_class(image_files)
        
        if not class_groups:
            logger.error("No valid image-annotation pairs found!")
            return
        
        logger.info(f"Found {len(class_groups)} classes")
        for class_id, samples in sorted(class_groups.items()):
            logger.info(f"  Class {class_id}: {len(samples)} samples")
        
        # Balance classes if enabled
        if self.should_balance_classes:
            class_groups = self.balance_classes(class_groups)
        
        # Create splits
        logger.info("\nCreating train/test/valid splits...")
        split_ratios = {
            'train': self.train_ratio,
            'test': self.test_ratio,
            'valid': self.valid_ratio
        }
        splits = self.create_splits(class_groups, split_ratios)
        
        logger.info("\nSplit summary:")
        for split_name, samples in splits.items():
            logger.info(f"  {split_name}: {len(samples)} samples")
        
        # Copy files
        logger.info("\nCopying files to dataset directory...")
        self.copy_files(splits)
        
        # Create data.yaml
        logger.info("\nCreating data.yaml configuration...")
        yaml_path = self.create_data_yaml(class_groups)
        
        logger.info("\n" + "=" * 80)
        logger.info("Dataset creation complete!")
        logger.info("=" * 80)
        logger.info(f"\nDataset location: {self.output_dir.absolute()}")
        logger.info(f"Configuration: {yaml_path}")
        logger.info("\nNext steps:")
        logger.info(f"  1. Review the dataset structure in: {self.output_dir}")
        logger.info(f"  2. Update data.yaml if needed to add custom class names")
        logger.info(f"  3. Start YOLO training with: yolo train data={yaml_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create balanced YOLO dataset with train/test/valid splits"
    )
    parser.add_argument(
        '--images',
        type=str,
        default='extracted_frames',
        help='Directory containing images (default: extracted_frames)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='yolo_annotations',
        help='Directory containing YOLO annotation files (default: yolo_annotations)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset',
        help='Output directory for dataset (default: dataset)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8 - industry standard)'
    )
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1 - industry standard)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1 - industry standard)'
    )
    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Disable class balancing (keep all samples)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = YOLODatasetBuilder(
        images_dir=args.images,
        annotations_dir=args.annotations,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
        balance_classes=not args.no_balance,
        seed=args.seed
    )
    
    builder.build()


if __name__ == '__main__':
    main()
