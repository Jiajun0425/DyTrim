# Learning Dynamics of Logits Debiasing for Long-Tailed Semi-Supervised Learning (ICLR2026)

This repository contains the official implementation of the methods proposed in our ICLR 2026 paper: [**Learning Dynamics of Logits Debiasing for Long-Tailed Semi-Supervised Learning**](https://jiajun0425.github.io/DyTrim/)

[Yue Cheng<sup>*1, 2</sup>](https://cyue0316.github.io/), [Jiajun Zhang<sup>*1</sup>](https://jiajun0425.github.io/), [Xiaohui Gao<sup>3</sup>](https://github.com/Gaitxh), [Weiwei Xing<sup>&#9993;1</sup>](https://faculty.bjtu.edu.cn/rjxy/7930.html), [Zhanxing Zhu<sup>&#9993;4</sup>](https://zhanxingzhu.github.io/)  

<sup>*</sup>Equal Contribution, <sup>&#9993;</sup>Corresponding Author  
<sup>1</sup>Beijing Jiaotong University <sup>2</sup>Ant Group
<sup>3</sup>Northwest Polytechnical University <sup>4</sup>University of Southampton

---

## 🧠 Method

<img width="825" alt="image" src="./figures/pipeline.png" />

**(I) Learning Dynamics Analysis.**
We analyze long-tailed semi-supervised learning from the perspective of learning dynamics and show that class imbalance induces accumulated logit bias that dominates model predictions.

**(II) Baseline Image as Bias Indicator.**
We introduce a task-irrelevant baseline image and theoretically show that its logits converge to the class prior, providing a direct and interpretable indicator of accumulated class bias.

**(III) Unified View of Debiasing Methods.**
Within this framework, existing debiasing strategies such as logit adjustment, reweighting, and resampling are unified as mechanisms that reshape gradient dynamics to counteract bias accumulation.

**(IV) Dynamic Dataset Pruning for LTSSL.**
Based on these insights, we propose DyTrim, a dynamic dataset pruning framework reallocates gradient budget via class-aware pruning on labeled data and confidence-based soft pruning on unlabeled data, outperforming existing state-of-the-art baselines.

---

## ⚙️ Environment

```bash
conda create -n dytrim python=3.10
conda activate dytrim
git clone https://github.com/Jiajun0425/DyTrim.git
cd DyTrim
pip install -r requirements.txt
```
> ⚠️ Note: Please use `pip install git+https://github.com/ildoonet/pytorch-randaugment` to install RandAugment.

---

## 📊 Datasets

If you want to use Small-ImageNet-127, follow the instructions in [prepare_small_imagenet_127/README.md](./prepare_small_imagenet_127/README.md).

---

## 🚀 Training

Below are the scripts for training the model on different datasets.

CIFAR-10:

```bash
python train.py --num_max 1500 --num_max_u 3000 --imb_ratio 100 --imb_ratio_u 100 --dataset cifar10 --gpu 0 --manualSeed 0
```

CIFAR-100:

```bash
python train.py --num_max 150 --num_max_u 300 --imb_ratio 50 --imb_ratio_u 50 --dataset cifar100 --gpu 0 --manualSeed 0
```

STL-10:

```bash
python train.py --num_max 450 --num_max_u 1 --imb_ratio 10 --imb_ratio_u 1 --dataset stl10 --gpu 0 --manualSeed 0
```

Small-ImageNet-127:

```bash
python train.py --img-size 32 --imb_ratio 1 --imb_ratio_u 1 --dataset smallimagenet --gpu 0 --manualSeed 0
```

---

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{cheng2026dytrim,
  title     = {Learning Dynamics of Logits Debiasing for Long-Tailed Semi-Supervised Learning},
  author    = {Cheng, Yue and Zhang, Jiajun and Gao, Xiaohui and Xing, Weiwei and Zhu, Zhanxing},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

## 🙏 Acknowledgements

Our code is based on [CDMAD](https://github.com/LeeHyuck/CDMAD) and [InfoBatch](https://github.com/NUS-HPC-AI-Lab/InfoBatch).

---

## 📬 Contact

For questions or issues, please:

- Open an issue on GitHub, or
- Contact: `jiajunzhang@bjtu.edu.cn`

---
