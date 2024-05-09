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
pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip3 install --no-cache-dir xformers==0.0.22

echo "Installing A1111 Web UI"
wget https://raw.githubusercontent.com/antonioglass/runpod-worker-a1111/main/install-automatic.py
python -m install-automatic --skip-torch-cuda-test

echo "Cloning ControlNet extension repo"
cd /workspace/stable-diffusion-webui
git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

echo "Cloning the ReActor extension repo"
git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor

echo "Cloning a person mask generator extension repo"
git clone --depth=1 https://github.com/djbielejeski/a-person-mask-generator.git extensions/a-person-mask-generator

echo "Installing dependencies for ControlNet"
cd /workspace/stable-diffusion-webui/extensions/sd-webui-controlnet
pip3 install -r requirements.txt

echo "Installing dependencies for ReActor"
cd /workspace/stable-diffusion-webui/extensions/sd-webui-reactor
pip3 install protobuf==3.20.3
pip3 install -r requirements.txt
pip3 install onnxruntime-gpu==1.16.3

echo "Installing dependencies for ControlNet"
cd /workspace/stable-diffusion-webui/extensions/a-person-mask-generator
pip3 install -r requirements.txt

echo "Installing the model for ReActor"
mkdir -p /workspace/stable-diffusion-webui/models/insightface
cd /workspace/stable-diffusion-webui/models/insightface
wget https://huggingface.co/antonioglass/reactor/resolve/main/inswapper_128.onnx

echo "Configuring ReActor to use the GPU instead of CPU"
echo "CUDA" > /workspace/stable-diffusion-webui/extensions/sd-webui-reactor/last_device.txt

echo "Installing the model for a person mask generator"
mkdir -p /workspace/stable-diffusion-webui/models/mediapipe
cd /workspace/stable-diffusion-webui/models/mediapipe
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite

echo "Installing RunPod Serverless dependencies"
cd /workspace/stable-diffusion-webui
pip3 install huggingface_hub runpod

echo "Downloading Stable Diffusion models"
# wget  --header="Authorization: Bearer HF_TOKEN"
cd /workspace/stable-diffusion-webui/models/Stable-diffusion
wget https://huggingface.co/antonioglass/models/resolve/main/3dAnimationDiffusion_v10.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/epicphotogasm_y.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/general_v3.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/meinahentai_v4.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/semi-realistic_v6.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/Deliberate_v3-inpainting.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/dreamshaper_631Inpainting.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/epicphotogasm_z-inpainting.safetensors
wget https://huggingface.co/antonioglass/models/resolve/main/meinahentai_v4-inpainting.safetensors

echo "Downloading ControlNet models"
mkdir -p /workspace/stable-diffusion-webui/models/ControlNet
cd /workspace/stable-diffusion-webui/models/ControlNet
wget https://huggingface.co/antonioglass/controlnet/resolve/main/controlnet11Models_openpose.safetensors
wget https://huggingface.co/antonioglass/controlnet/raw/main/controlnet11Models_openpose.yaml
wget https://huggingface.co/antonioglass/controlnet/resolve/main/control_v11p_sd15_inpaint.pth
wget https://huggingface.co/antonioglass/controlnet/raw/main/controlnet11Models_openpose.yaml

echo "Downloading Upscalers"
mkdir -p /workspace/stable-diffusion-webui/models/ESRGAN
cd /workspace/stable-diffusion-webui/models/ESRGAN
wget https://huggingface.co/antonioglass/upscalers/resolve/main/4x-AnimeSharp.pth
wget https://huggingface.co/antonioglass/upscalers/resolve/main/4x_NMKD-Siax_200k.pth
wget https://huggingface.co/antonioglass/upscalers/resolve/main/8x_NMKD-Superscale_150000_G.pth

echo "Downloading Embeddings"
mkdir -p /workspace/stable-diffusion-webui/embeddings
cd /workspace/stable-diffusion-webui/embeddings
wget https://huggingface.co/antonioglass/embeddings/resolve/main/BadDream.pt
wget https://huggingface.co/antonioglass/embeddings/resolve/main/FastNegativeV2.pt
wget https://huggingface.co/antonioglass/embeddings/resolve/main/UnrealisticDream.pt

echo "Downloading Loras"
mkdir -p /workspace/stable-diffusion-webui/models/Lora
cd /workspace/stable-diffusion-webui/models/Lora
wget https://huggingface.co/antonioglass/loras/resolve/main/EkuneCowgirl.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/EkunePOVFellatioV2.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/EkuneSideDoggy.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/IPV1.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/JackOPoseFront.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/LickingOralLoRA.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/POVAssGrab.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/POVDoggy.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/POVMissionary.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/POVPaizuri.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/POVReverseCowgirl.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/PSCowgirl.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/RSCongress.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/SelfBreastGrab.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/SideFellatio.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/TheMating.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/cuddling_handjob_v0.1b.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/hand_in_panties_v0.82.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/jkSmallBreastsLite_V01.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/masturbation_female.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/shirtliftv1.safetensors
wget https://huggingface.co/antonioglass/loras/resolve/main/yamato_v2.safetensors

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
