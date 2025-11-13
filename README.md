
# Video Frame Extraction and Auto-Annotation Pipeline

A modular Python pipeline for extracting frames from videos and automatically generating YOLO-format annotations using Grounding DINO for object detection.

## 🎯 Features

- **Frame Extraction**: Extract frames from videos at configurable intervals
- **Auto-Annotation**: Use Grounding DINO for zero-shot object detection
- **YOLO Format**: Generates annotations in YOLO format ready for training
- **Visualization**: Creates annotated images to verify detections
- **Modular Design**: Easy to integrate into existing data pipelines
- **Flexible Configuration**: YAML-based config with CLI overrides

## 📋 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Grounding DINO
- PIL/Pillow
- PyYAML

## 🚀 Installation

### 1. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 2. Install Grounding DINO

```powershell
# Clone Grounding DINO repository
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Download Grounding DINO model weights

```powershell
# Create weights directory
mkdir weights

# Download model weights
# Visit: https://github.com/IDEA-Research/GroundingDINO/releases
# Download: groundingdino_swint_ogc.pth
# Place it in the weights/ directory
```

## 📁 Project Structure

```
training/
├── main.py                          # Main pipeline script
├── frame_extractor.py              # Video frame extraction module
├── grounding_dino_annotator.py     # Grounding DINO detection module
├── yolo_formatter.py               # YOLO format converter
├── visualizer.py                   # Bounding box visualization
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── raw_footages/                   # Input videos directory
├── extracted_frames/               # Output frames (auto-created)
├── yolo_annotations/               # Output YOLO labels (auto-created)
└── visualizations/                 # Output visualizations (auto-created)
```

## ⚙️ Configuration

Create or edit `config.yaml`:

```yaml
paths:
  video_input: raw_footages
  frames_output: extracted_frames
  annotations_output: yolo_annotations
  visualizations_output: visualizations

extraction:
  frame_interval: 30              # Extract every 30th frame
  max_frames_per_video: null      # null = extract all frames
  target_size: null               # null = keep original, or [width, height]

grounding_dino:
  config_path: null               # null = use default
  checkpoint_path: null           # null = use default
  text_prompt: "person . car . object"  # Customize for your objects
  box_threshold: 0.35             # Detection confidence threshold
  text_threshold: 0.25            # Text matching threshold
  min_confidence: 0.3             # Minimum confidence to save

visualization:
  enabled: true
  show_confidence: true
  max_images: 50                  # Max images to visualize
  create_grid: true               # Create grid of samples
```

## 🎮 Usage

### Basic Usage

```powershell
# Run with default config.yaml
python main.py

# Specify custom config
python main.py --config my_config.yaml
```

### CLI Overrides

```powershell
# Override video directory
python main.py --video-dir raw_footages

# Override detection prompt
python main.py --prompt "lesion . tumor . abnormality"

# Override frame interval
python main.py --frame-interval 15

# Combine multiple overrides
python main.py --video-dir videos --prompt "person . vehicle" --frame-interval 10
```

## 🔧 Advanced Usage

### Custom Detection Prompts

For medical imaging:
```yaml
text_prompt: "lesion . tumor . abnormality . nodule . mass . cyst"
```

For traffic detection:
```yaml
text_prompt: "person . car . bicycle . motorcycle . bus . truck . traffic light"
```

For general objects:
```yaml
text_prompt: "object . person . animal . vehicle"
```

### Frame Extraction Settings

Extract every frame (high quality, large dataset):
```yaml
frame_interval: 1
```

Extract every 30th frame (1 frame per second at 30fps):
```yaml
frame_interval: 30
```

Limit frames per video:
```yaml
max_frames_per_video: 100
```

Resize frames:
```yaml
target_size: [1280, 720]  # [width, height]
```

### Integration into Existing Pipelines

Use individual modules:

```python
from frame_extractor import VideoFrameExtractor
from grounding_dino_annotator import GroundingDINOAnnotator
from yolo_formatter import YOLOFormatter
from visualizer import BoundingBoxVisualizer

# Extract frames
extractor = VideoFrameExtractor(output_dir="frames")
frame_paths = extractor.extract_frames("video.mp4", frame_interval=30)

# Annotate with Grounding DINO
annotator = GroundingDINOAnnotator()
boxes, labels, scores = annotator.annotate_image(
    frame_paths[0],
    "person . car"
)

# Convert to YOLO format
formatter = YOLOFormatter(output_dir="labels")
formatter.save_annotation(
    frame_paths[0],
    boxes,
    labels,
    image_width=1920,
    image_height=1080
)

# Visualize
visualizer = BoundingBoxVisualizer(output_dir="vis")
visualizer.draw_boxes(frame_paths[0], boxes, labels, scores)
```

## 📊 Output Format

### YOLO Annotation Format

Each frame gets a `.txt` file with the same name:
```
1_0.jpg -> 1_0.txt
```

Annotation format (one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).

### Directory Structure After Running

```
extracted_frames/
├── 1/
│   ├── 1_0.jpg
│   ├── 1_30.jpg
│   └── 1_60.jpg
├── 2/
│   ├── 2_0.jpg
│   └── 2_30.jpg

yolo_annotations/
├── 1_0.txt
├── 1_30.txt
├── 1_60.txt
├── 2_0.txt
├── 2_30.txt
├── classes.txt
└── dataset.yaml

visualizations/
├── 1_0_annotated.jpg
├── 1_30_annotated.jpg
└── grid_visualization.jpg
```

## 🎓 Training with YOLO

After running the pipeline, use the generated dataset:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='yolo_annotations/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 🐛 Troubleshooting

### Grounding DINO not found
```powershell
pip install groundingdino-py
# Or install from source as shown in Installation section
```

### CUDA out of memory
- Reduce batch size in annotation
- Process images in smaller batches
- Use CPU instead: Edit config to use CPU device

### No detections
- Adjust `box_threshold` and `text_threshold` (lower values = more detections)
- Check your `text_prompt` matches objects in videos
- Verify Grounding DINO model is loaded correctly

### Frame extraction slow
- Increase `frame_interval` to extract fewer frames
- Set `max_frames_per_video` to limit frames
- Use smaller `target_size` for resize

## 📝 Notes

- Frame filenames follow format: `{video_name}_{frame_number}.jpg`
- Grounding DINO supports zero-shot detection (no training needed)
- Adjust detection thresholds based on your use case
- Review visualizations before starting YOLO training
- The pipeline is designed for easy integration into larger workflows

## 🤝 Contributing

This is a modular pipeline designed for flexibility. Feel free to:
- Add new annotation methods
- Implement additional formatters (COCO, VOC, etc.)
- Extend visualization capabilities
- Add preprocessing steps

## 📄 License

MIT License - Feel free to use and modify for your projects.
