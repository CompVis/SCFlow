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
wget -O scflow_last.ckpt https://huggingface.co/CompVis/SCFlow/resolve/main/scflow_last.ckpt?dowload=true

# unclip checkpoint for visualization
wget -O sd21-unclip-l.ckpt https://huggingface.co/CompVis/SCFlow/resolve/main/sd21-unclip-l.ckpt?dowload=true
```
## 🔥 Usage
Inference forward (merge content and style)
```bash
bash scripts/inference_forward.sh
```
Inference reverse (disentangle content and style from a given reference)
```bash
bash scripts/inference_reverse.sh
```

Training (coming soon)
```bash
bash ...
```

## 🗂️ Dataset
Coming soon

## 🎓 Citation


If you use this codebase or otherwise found our work valuable, please cite our paper:
```bibtex
@article{ma2025scflow,
  title={SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models},
  author={Ma, Pingchuan and Yang, Xiaopei and Li, Yusong and Gui, Ming and Krause, Felix and Schusterbauer, Johannes and Ommer, Bj{\"o}rn},
  journal={arXiv preprint arXiv:2508.03402},
  year={2025}
}
```

## 🔥 Updates and Backlogs
- [x] **[06.08.2025]** [ArXiv](https://arxiv.org/abs/2508.03402) paper avaiable.
- [x] **[12.08.2025]** Release Inference code and ckpt
- [ ] Host the dataset and training code
