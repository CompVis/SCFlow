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


🔥 News
* [06.2026] This repository now also contains the official code for the paper: **catFM: Contrastive-Augmented Flow Matching for
 Style-Content Disentanglement,** a follow-up work currently under review at TPAMI.
* [10.2025] Released training code and dataset splits.
* [10.2025] Released the full 512px image dataset.
* [08.2025] Released inference code and pretrained checkpoints.
* [08.2025] ICCV paper available on arXiv.

[!IMPORTANT]
The original SCFlow (ICCV 2025) implementation remains the default training and inference pipeline. This repository additionally includes the implementation of catFM, a follow-up method currently under review at TPAMI.

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

### 🐈 catFM Follow-up
This repository additionally includes the implementation of catFM, a follow-up work built upon SCFlow and currently under review at TPAMI.

Compared to SCFlow, CATFM introduces:

* contrastive regularization on style and content embeddings,
* multiple endpoint prediction objectives,
* improved style-content disentanglement and retrieval performance.

The original SCFlow pipeline remains the default. To train CATFM, use:

```bash
bash scripts/catfm_training.sh
```

You can also customize the training configuration directly from the command line:

```bash
python training.py --config configs/catfm_training.yaml train.dml_type=MultiSimilarity train.predict_x0x1=True
```

For catFM metric losses (`train.dml_type != null`), install the optional dependency:
```bash
pip install pytorch-metric-learning
```

catFM checkpoints can be used by the same inference script:
```bash
python inference.py \
    --model_type catfm \
    --config configs/inference.yaml \
    --resume_checkpoint path/to/catfm.ckpt \
    --image_c_path path/to/content.jpg \
    --image_s_path path/to/style.jpg \
    --unclip_ckpt ckpts/sd21-unclip-l.ckpt
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

### Original Images in 512px
We hosted the original images on HF. You should be able to download them by calling:
```bash

# The zip file is around 36.5 GB. 
wget https://huggingface.co/CompVis/SCFlow/resolve/main/raw_512px.zip

```
It is structured by styles, then different content ids, e.g., `Cubism/00001.jpg ... 10000.jpg`, where the content ids are consistent across different styles.

## 🎓 Citation & Contact

If you use this codebase and dataset, or found our work valuable, please cite our paper:
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