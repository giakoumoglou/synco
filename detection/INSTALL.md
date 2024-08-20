# Detectron2 Installation Guide

## Installation Steps

1. Load Anaconda module:
```
module load anaconda3/personal
```

2. Create conda environment:
```
conda create -n detectron2_cuda10 -c conda-forge cudatoolkit=10.2 python=3.9
```

3. Activate environment:
```
source ~/.bashrc
conda activate detectron2_cuda10
```

4. Install NVIDIA cuDNN:
```
python3 -m pip install nvidia-cudnn-cu10==7.6.5.32
```

5. Install PyTorch 1.7:
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

6. Install OpenCV:
```
pip install opencv-python
```

7. Install Cython and PyYAML:
```
pip install cython pyyaml==5.1
```

8. Install COCO API:
```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

9. Install specific versions of NumPy and Pillow:
```
pip install numpy==1.23.5
pip install pillow==9.1.0
```

10. Download Detectron2 v0.5 source code from https://github.com/facebookresearch/detectron2/releases

11. Install Detectron2:
```
python -m pip install -e detectron2
```

## Verification

To verify the installation and check versions, run the following Python script:

```python
import torch
import detectron2
import torch.cuda

print(f"PyTorch version: {torch.__version__}")
print(f"Detectron2 version: {detectron2.__version__}")
print(f"CUDA version: {torch.version.cuda}")
```
