"""
Frame Extractor Module
Handles video frame extraction with configurable parameters.
"""
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from video files at specified intervals."""
    
    def __init__(self, output_dir: str = "extracted_frames"):
        """
        Initialize the frame extractor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(
        self,
        video_path: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> list:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            frame_interval: Extract every Nth frame (1 = every frame)
            max_frames: Maximum number of frames to extract (None = all)
            target_size: Resize frames to (width, height) if specified
            
        Returns:
            List of extracted frame paths
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create subdirectory for this video
        video_name = video_path.stem
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {video_name}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        extracted_paths = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                if max_frames and extracted_count >= max_frames:
                    break
                
                # Resize if specified
                if target_size:
                    frame = cv2.resize(frame, target_size)
                
                # Save frame with suffix _frameNumber
                frame_filename = f"{video_name}_{frame_count}.jpg"
                frame_path = video_output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_paths.append(frame_path)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    logger.info(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from {video_name}")
        
        return extracted_paths
    
    def extract_from_directory(
        self,
        video_dir: str,
        video_extensions: list = ['.mp4', '.avi', '.mov', '.mkv'],
        **kwargs
    ) -> dict:
        """
        Extract frames from all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            video_extensions: List of video file extensions to process
            **kwargs: Additional arguments passed to extract_frames()
            
        Returns:
            Dictionary mapping video names to lists of extracted frame paths
        """
        video_dir = Path(video_dir)
        results = {}
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        video_files = sorted(video_files)
        logger.info(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            try:
                frame_paths = self.extract_frames(str(video_file), **kwargs)
                results[video_file.stem] = frame_paths
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {str(e)}")
        
        return results
