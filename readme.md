# RAFT-MSF++ （Under Review）

This repository provides the official PyTorch implementation of:

> **RAFT-MSF++: Temporal Geometry-Motion Feature Fusion for Self-Supervised Monocular Scene Flow**

---

## Environment

<!-- ### PyTorch Installation -->

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Tested Environment

* **Python**: 3.10.18
* **PyTorch**: 2.7.0
* **TorchVision**: 0.22.0
* **TorchAudio**: 2.7.0


In practice, you can also use Python 3.7, PyTorch 1.12.1, torchvision 0.13.1, and torchaudio 0.12.1. There are no strict requirements for the Python or PyTorch versions, but be aware of modifications in `forwardwarp_package/setup.py`. Additionally, `softsplat` may have better compatibility with lower PyTorch versions (CuPy 11.6.0). Installing PyTorch 2.0 or higher is mainly to satisfy the `FlashAttention` dependency required by MemFlow.

<!-- ### CUDA & GPU Support

* **CUDA Toolkit**: CUDA 12.x
* **cuDNN**: 9.5.1
* **NCCL**: 2.26.2
* PyTorch installed with **CUDA 12 support** (`torch==2.7.0+cu12x`)

### Key Dependencies

* `timm` 1.0.15
* `numpy` 1.24.4
* `opencv-python` 4.11.0
* `scipy` 1.15.3
* `scikit-image` 0.25.2
* `scikit-learn` 1.7.2
* `matplotlib` 3.10.1

--- -->

## Installation and Dataset

For installation details and dataset configuration, please refer to:

* [Self-Supervised Monocular Scene Flow Estimation](https://github.com/visinf/self-mono-sf)

### Semantic Segmentation and Occlusion Regularization

For semantic segmentation and the **Occlusion Regularization loss**, we follow the official **Segment Anything Model (SAM)** repository to generate SAM masks for all samples. Then, modify the `sam_fullseg_root` path in `datasets/kitti_raw_monosf.py  datasets/kitti_2015_train.py` to point to your own directory.

Relevant implementation can be found in:

```
datasets/sam_kitti_raw_sf.py
```

---

## Training and Inference

### Training

To train the model, run:

```bash
sh scripts/my_scripts/train_selfsup.sh
```

### Evaluation (Pretrained RAFT-MSF++)

Evaluate on the KITTI training set:

```bash
sh scripts/my_scripts/eval_monosf_selfsup_kitti_train.sh
```

Evaluate on the KITTI test set:

```bash
sh scripts/my_scripts/eval_monosf_selfsup_kitti_test.sh
```

### Ablation Studies

All ablation experiments are provided in:

```
scripts/my_scripts/ablation/
```

For each table reported in the paper, a corresponding script is included.

---

## Pretrained Models

Pretrained checkpoints are available in:

```
ckpt/
```

---

## Acknowledgement

<!-- If you use this codebase in your research, please cite our paper:

```bibtex
@article{Bayramli2022RAFTMSFSM,
  title={RAFT-MSF: Self-Supervised Monocular Scene Flow using Recurrent Optimizer},
  author={Bayram Bayramli and Junhwa Hur and Hongtao Lu},
  journal={International Journal of Computer Vision},
  year={2023},
  url={https://doi.org/10.1007/s11263-023-01828-4}
}
``` -->

This codebase is adapted from the following projects:

* [Self-MONO-SF](https://github.com/visinf/self-mono-sf)
* [RAFT](https://github.com/princeton-vl/RAFT)
* [RAFT-MSF](https://github.com/Bayrambai/raft-msf)

We sincerely thank the authors for their valuable contributions.
