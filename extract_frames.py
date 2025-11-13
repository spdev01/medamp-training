"""
Quick Start Script - Frame Extraction Only
Extract frames from videos without requiring Grounding DINO installation.
"""
import argparse
from pathlib import Path
import logging

from frame_extractor import VideoFrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Quick start for frame extraction only."""
    parser = argparse.ArgumentParser(
        description="Quick frame extraction from videos"
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='raw_footages',
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='extracted_frames',
        help='Output directory for frames'
    )
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=30,
        help='Extract every Nth frame (default: 30)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames per video (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Frame Extraction Quick Start")
    logger.info("=" * 80)
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Frame interval: {args.frame_interval}")
    logger.info(f"Max frames per video: {args.max_frames if args.max_frames else 'unlimited'}")
    
    # Extract frames
    extractor = VideoFrameExtractor(output_dir=args.output_dir)
    results = extractor.extract_from_directory(
        video_dir=args.video_dir,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames
    )
    
    # Summary
    total_frames = sum(len(paths) for paths in results.values())
    logger.info("\n" + "=" * 80)
    logger.info("Extraction Complete!")
    logger.info("=" * 80)
    logger.info(f"Processed {len(results)} videos")
    logger.info(f"Extracted {total_frames} frames total")
    logger.info(f"Frames saved to: {args.output_dir}")
    logger.info("\nNext: Install Grounding DINO to auto-annotate these frames")
    logger.info("Or manually annotate frames using labeling tools like LabelImg")


if __name__ == '__main__':
    main()
