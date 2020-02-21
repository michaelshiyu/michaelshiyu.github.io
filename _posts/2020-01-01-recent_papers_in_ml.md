---
title: 'Recent Papers in Machine Learning'
date: 2020-01-01
permalink: /posts/2020/01/blog-post-1/
tags:
  - research
---

<!---
[Template]()
======

**keywords:** 

**code:** 

**datasets:** 

**one-sentence summary:** 

**details on main method:** 

**additional comments:** 
)
-->

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
======

**keywords:** semi-supervised learning, image classification

**code:** [available](https://github.com/google-research/fixmatch)

**datasets:** CIFAR-10/100, SVHN, STL-10, ImageNet

**one-sentence summary:** Combines two existing semi-supervised learning techniques, consistency regularization and pseudo-labeling, in a simplistic way but achieves state-of-the-art results (in particular, 94.93%/88.61% on CIFAR-10 with 250/40 labels). 

**details on main method:** Is essentially a loss function of two terms: cross-entropy on weakly augmented labeled examples (Eq. 3) and cross-entropy on strongly augmented, unlabeled examples using (only confident enough) model predictions (on weakly augmented versions of these images) as artificial labels (Eq. 4). 

**additional comments:** 
- Nice literature review. 
- Very detailed descriptions on the experimental settings. 
- Extensive ablation study.

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
======

**keywords:** self-supervised learning, image classification 

**code:** 

**datasets:** self-supervised, semi-supervised evaluated on: ImageNet; transfer learning evaluated on: Food, CIFAR-10/100, Birdsnap, SUN397, Cars, Aircraft, VOC2007, DTD, Pets, Caltech-101, Flowers  

**one-sentence summary:** Based on contrastive learning, the authors proposed a simple self-supervised representation learning framework and demonstrated an instantiation with an off-the-shelf ResNet-50 backbone that achieved state-of-the-art performance on ImageNet (76.5% top-1 self-supervised, 85.8% top-5 fine-tuned with 1% of the labels). 

**details on main method:** Fig. 2 and Alg. 1 together gives a clear description. The particular contrastive loss used (NT-Xent) is in Eq. 1. Two augmented versions of the same image form a positive pair in the contrastive loss, whereas the negative pairs are simply pairs of augmented versions of distinct images. The contrastive loss is computed after a trainable projection head (discarded after training) that projects the representation into a low-dimensional space. 

**additional comments:** 
- Evaluation methods used: 
    - Linear evaluation (train a supervised linear classifier on top of the learned representations). Main results in Table 6;
    - Fine-tuned with few labels (semi-supervised). Main results in Table 7; 
    - Transfer learning (froze or fine-tuned the self-supervised components). Main results in Table 8. 
- Nice literature review.
- Somewhat unusual section names: it seems that all sections starting from Sec. 3 are experiments and ablation study. 
- Extensive ablation study.
- The 1x, 2x, and 4x denote the width multiplier for the ResNet-50. This is described in the beginning of Sec. 6. 
