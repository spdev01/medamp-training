#!/bin/bash
# Medical Ampoule Detection Training Environment Setup

set -e  # Exit on error

echo "=========================================="
echo "Medical Ampoule Detection Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# 1. Update system packages
echo ""
echo "Step 1: Updating system packages..."
apt-get update -qq
apt-get install -y zip unzip git wget curl screen tmux htop -qq
print_status "System packages updated"

# 2. Check Python and CUDA
echo ""
echo "Step 2: Checking Python and CUDA..."
python_version=$(python --version 2>&1)
print_status "Python: $python_version"

if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//' 2>/dev/null || echo "N/A")
    print_status "GPU: $gpu_info"
    print_status "CUDA: $cuda_version"
else
    print_warning "NVIDIA GPU not detected (CPU mode)"
fi

# 3. Install Python dependencies
echo ""
echo "Step 3: Installing Python packages..."
pip install -q --upgrade pip
pip install -q ultralytics
pip install -q opencv-python
pip install -q PyYAML
pip install -q pillow
pip install -q numpy
pip install -q tqdm
print_status "Python packages installed"

# 4. Verify PyTorch and CUDA
echo ""
echo "Step 4: Verifying PyTorch installation..."
python << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PYEOF
print_status "PyTorch verified"

# 5. Clone Grounding DINO repository
echo ""
echo "Step 5: Setting up Grounding DINO..."
if [ -d "GroundingDINO" ]; then
    print_warning "GroundingDINO directory already exists. Skipping clone."
else
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    print_status "Grounding DINO cloned"
fi

# 6. Download Grounding DINO weights
echo ""
echo "Step 6: Downloading Grounding DINO weights..."
mkdir -p weights
cd weights

if [ -f "groundingdino_swinb_cogcoor.pth" ]; then
    print_warning "Grounding DINO weights already exist. Skipping download."
else
    echo "Downloading Grounding DINO Swin-B weights..."
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    print_status "Grounding DINO weights downloaded"
fi
cd ..

# 7. Download YOLO model weights
echo ""
echo "Step 7: Downloading YOLO model weights..."
read -p "Download YOLO model weights? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p yolo_models
    cd yolo_models
    
    # Array of model URLs
    declare -a models=(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt"
    )
    
    echo ""
    echo "Downloading YOLO models to: $(pwd)"
    echo ""
    
    for model_url in "${models[@]}"; do
        model_name=$(basename "$model_url")
        
        if [ -f "$model_name" ]; then
            print_warning "$model_name already exists. Skipping."
        else
            echo "Downloading $model_name..."
            wget -q --show-progress "$model_url"
            print_status "$model_name downloaded"
        fi
    done
    
    echo ""
    print_status "All YOLO models downloaded successfully!"
    echo ""
    echo "Available models:"
    ls -lh *.pt
    
    cd ..
else
    print_warning "Skipping YOLO model download."
fi

# 8. Create project directory structure
echo ""
echo "Step 8: Creating project directories..."
mkdir -p extracted_frames
mkdir -p yolo_annotations
mkdir -p visualizations
mkdir -p dataset/train/images
mkdir -p dataset/train/labels
mkdir -p dataset/test/images
mkdir -p dataset/test/labels
mkdir -p dataset/valid/images
mkdir -p dataset/valid/labels
mkdir -p runs/detect
print_status "Project directories created"

# 9. Display next steps
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_status "Working directory: $(pwd)"
echo ""
echo "Next steps:"
echo ""
echo "1. Upload your video files to raw_footages/"
echo ""
echo "2. Extract frames and annotate:"
echo "   python main.py"
echo ""
echo "3. Create balanced dataset:"
echo "   python create_dataset.py"
echo ""
echo "4. Train YOLO model:"
echo "   screen -S training"
echo "   yolo train data=dataset/data.yaml model=yolo_models/yolo12m.pt epochs=100 imgsz=640"
echo ""
echo "   Detach from screen: Ctrl+A, then D"
echo "   Reattach to screen: screen -r training"
echo ""
echo "=========================================="