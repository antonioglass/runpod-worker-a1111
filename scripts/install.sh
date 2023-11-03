#!/usr/bin/env bash
echo "Deleting Automatic1111 Web UI"
rm -rf /workspace/stable-diffusion-webui

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning A1111 repo to /workspace"
cd /workspace
git clone --depth=1 https://github.com/antonioglass/stable-diffusion-webui.git

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Creating and activating venv"
cd stable-diffusion-webui
python -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip install --no-cache-dir xformers

echo "Installing A1111 Web UI"
wget https://raw.githubusercontent.com/antonioglass/runpod-worker-a1111/main/install-automatic.py
python -m install-automatic --skip-torch-cuda-test

echo "Cloning ControlNet extension repo"
git clone https://github.com/antonioglass/sd-webui-controlnet.git extensions/sd-webui-controlnet

echo "Installing dependencies for ControlNet"
cd extensions/sd-webui-controlnet
pip install -r requirements.txt

echo "Installing RunPod Serverless dependencies"
cd /workspace/stable-diffusion-webui
pip3 install huggingface_hub runpod>=0.10.0

echo "Downloading Stable Diffusion models"
cd /workspace/stable-diffusion-webui/models/Stable-diffusion
wget https://civitai.com/api/download/models/197181 --content-disposition

echo "Downloading ControlNet models"
mkdir -p /workspace/stable-diffusion-webui/models/ControlNet
cd /workspace/stable-diffusion-webui/models/ControlNet
wget https://civitai.com/api/download/models/44811 --content-disposition

echo "Creating log directory"
mkdir -p /workspace/logs

echo "Installing config files"
cd /workspace/stable-diffusion-webui
rm webui-user.sh config.json ui-config.json
wget https://raw.githubusercontent.com/antonioglass/runpod-worker-a1111/main/webui-user.sh
wget https://raw.githubusercontent.com/antonioglass/runpod-worker-a1111/main/config.json
wget https://raw.githubusercontent.com/antonioglass/runpod-worker-a1111/main/ui-config.json

echo "Starting A1111 Web UI"
deactivate
export HF_HOME="/workspace"
cd /workspace/stable-diffusion-webui
./webui.sh -f
