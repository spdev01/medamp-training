#!/bin/bash

# Clone Grounding DINO repository
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

# Install requirements
# Download Grounding DINO weights
mkdir -p weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ../..

echo "Grounding DINO setup completed!"