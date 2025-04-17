
# MUSIQ: Multi-Scale Image Quality Transformer (Fine-tuned on AVA)

This repository contains a PyTorch-based implementation of the MUSIQ model, adapted from the original paper:

> **MUSIQ: Multi-Scale Image Quality Transformer**  
> Ke Ma, Zhengfang Duanmu, Qifei Wang, Zhihao Hu, Yilin Wang, Yilin Li, C.-C. Jay Kuo, Zhi Li  
> [arXiv:2108.05997](https://arxiv.org/abs/2108.05997)

---

## 🔧 Environment

This code was tested with the following dependencies:

- `pytorch==1.7.1` (with CUDA 11.0)
- `einops==0.3.0`
- `numpy==1.18.3`
- `opencv-python==4.2.0`
- `scipy==1.4.1`
- `tqdm==4.45.0`
- `json==2.0.9`

---

## 🔍 Fine-tuning on AVA Dataset

We fine-tuned the MUSIQ model on the **AVA (Aesthetic Visual Analysis)** dataset, which provides aesthetic quality ratings for over 250,000 images.

### ✅ Modifications:

- The original ResNet50 backbone (pretrained on ImageNet) is retained.
- Input resolution is adapted to support AVA images (various sizes).
- The model is fine-tuned using aesthetic scores from AVA.
- We adjusted the regression head to better fit AVA’s aesthetic distribution.

---

## 📂 Model Setup

Before running inference, make sure you place the required model files correctly:

- Download **ResNet50 weights** from [here](https://download.pytorch.org/models/resnet50-0676ba61.pth)
  - Rename it as `resnet50.pth`
  - Place it in:  
    ```bash
    ../MUSIQ/model/resnet50.pth
    ```

- Ensure your fine-tuned checkpoint is named `epoch100.pth` and placed in:[here](https://drive.google.com/file/d/1SG_f9T4eVIyhRBVpGy8LlQiDCMwiHb0O/view?usp=sharing)
  ```bash
  ../MUSIQ/checkpoints/epoch100.pth
  ```

---

## 🚀 Inference

To run inference:

```bash
python inference.py
```

Edit the following variables inside `inference.py`:

- `dirname`: path to the folder of test images
- `checkpoint`: path to the trained model (e.g., `checkpoints/epoch100.pth`)
- `result_score_txt`: file path to save the output scores

---

## 📜 Citation

If you use this work or the fine-tuned model, please cite the original MUSIQ paper:

```bibtex
@article{ma2021musiq,
  title={MUSIQ: Multi-Scale Image Quality Transformer},
  author={Ma, Ke and Duanmu, Zhengfang and Wang, Qifei and Hu, Zhihao and Wang, Yilin and Li, Yilin and Kuo, C.-C. Jay and Li, Zhi},
  journal={arXiv preprint arXiv:2108.05997},
  year={2021}
}
```

---
