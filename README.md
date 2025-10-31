<p align="center">
 <h2 align="center">SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models</h2>
 <p align="center"> 
    Pingchuan Ma<sup>*</sup> · Xiaopei Yang<sup>*</sup> · Yusong Li
 </p><p align="center"> 
    Ming Gui · Felix Krause · Johannes Schusterbauer · Björn Ommer
 </p>
 <p align="center"> 
   <b>CompVis Group @ LMU Munich</b> &nbsp;&nbsp;&nbsp; <b>Munich Center for Machine Learning (MCML)</b> 
 </p>
 <p align="center"> 
    
 </p>
 <p align="center"> <sup>*</sup> <i>equal contribution</i> </p>
 
<p align="center"><strong>📄 ICCV 2025</strong></p>
  
</p>


<a href="https://compvis.github.io/SCFlow/"><img src="docs/static/figures/badge-website.svg" alt="Website"></a>
<a href="https://arxiv.org/abs/2508.03402"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" alt="Paper"></a>
<a href="https://huggingface.co/CompVis/SCFlow"><img src="https://img.shields.io/badge/HuggingFace-Weights-orange" alt="Paper"></a>

This repository contains the official implementation of the paper "SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models".
We proposed a flow-matching framework that learns an invertible mapping between style-content mixtures and their separate representations, avoiding explicit disentanglement objectives. Together with the method, we have curated a 510k synthetic dataset consisting of 10k content instances and 51 distinct styles.


<p align="center">
   <img src="docs/static/images/teaser.jpg" alt="Cover" width="80%">
</p>



## 🛠️ Setup
Create the enviroment with conda:
```bash
conda create -n scflow python=3.10
conda activate scflow
pip install -r requirements.txt
```
The enviroment was tested on `Ubuntu 22.04.5 LTS` with `CUDA 12.1`. You can *optionally* install jupyter-notebook to run the notebook provided in [`notebooks`](https://github.com/CompVis/SCFlow/tree/main/notebooks)

Download the model checkpoints:
```bash
mkdir ckpts
cd ckpts

# model checkpoint
wget https://huggingface.co/CompVis/SCFlow/resolve/main/scflow_last.ckpt

# unclip checkpoint for visualization
wget https://huggingface.co/CompVis/SCFlow/resolve/main/sd21-unclip-l.ckpt
```

Download the training and test splits of the dataset:
```bash
# return to parent dir
cd ..
mkdir dataset
cd dataset

# training split with meta data, e.g., content and style idx and content description etc.
wget https://huggingface.co/CompVis/SCFlow/resolve/main/train.h5


# test split with meta data, e.g., content and style idx and content description etc.
wget https://huggingface.co/CompVis/SCFlow/resolve/main/test.h5

```


## 🔥 Usage
The following bash scripts are just naive wrappers for an easy start. You can the args accordingly by calling directly the `training.py` and `inference.py`.

Inference forward (merge content and style)
```bash
bash scripts/inference_forward.sh
```
Inference reverse (disentangle content and style from a given reference)
```bash
bash scripts/inference_reverse.sh
```

For training you would need ~22GB with the default setting.
```bash
bash scripts/training.sh
```

## 🗂️ Dataset Overview
We hosted the dataset (currently only the clip embeddings and their corresponding metadata due to the space limit) on HF. You can download them as instructed in the above section. The file `train.h5` (same holds for `test.h5`) is an HDF5 dataset storing embeddings and metadata useful for training. You can load it in Python with:

```python
import h5py
train = h5py.File(”./dataset/train.h5”, ‘r’)
```

The main groups inside are:

- **images**: Contains CLIP embeddings with shape `(357000, 768)`, representing feature vectors for training samples.
- **metadata**: Contains descriptive information with keys:
  - `content_description`
  - `content_idx`
  - `style_idx`
  - `style_name`

> **Note:** Some metadata entries can be duplicated because there are 7000 content variations for training and 3000 for testing. This means the same content with different styles will have identical `content_description` and `content_idx`.



## 🎓 Citation & Contact

If you use this codebase or otherwise found our work valuable, please cite our paper:
```bibtex
@inproceedings{ma2025scflow,
    author    = {Ma, Pingchuan and Yang, Xiaopei and Li, Yusong and Gui, Ming and Krause, Felix and Schusterbauer, Johannes and Ommer, Bj\"orn},
    title     = {SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {14919-14929}
}
```

In case you encounter any issues or would like to collaborate, plz feel free to drop me a message:
* Email: p.ma(at)lmu(dot)de
* [linkedin](https://www.linkedin.com/in/pingchuan-ma-492543156/)

## 🔥 Updates and Backlogs
- [x] **[06.08.2025]** [ArXiv](https://arxiv.org/abs/2508.03402) paper avaiable.
- [x] **[12.08.2025]** Release Inference code and ckpt.
- [x] **[31.10.2025]** Host the dataset (latent and meta data) and training code.
- [ ] We are working on a solution to host the original images.
