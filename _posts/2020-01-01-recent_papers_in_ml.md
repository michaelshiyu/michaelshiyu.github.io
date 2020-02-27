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

[AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485)
======

**keywords:** image generation conditioned on text 

**code:** 

**datasets:** CUB, COCO 

**evaluation metric(s):** inception score, R-precision (to quantify how well a generated image matches the given text. Discussed in detail in the beginning of Sec. 4)

**one-sentence summary:** 
Existing text-to-image translation methods encode all text description into a single feature vector, lacking fine-grained, word-level information. 
This work encodes each word into a feature vector (in addition to the sentence feature vector) and utilizes attention mechanism to enable the network to generate regions based on word(s) that are most relevant to it. 
The authors also proposed an attention-based similarity model that quantifies the similarity between the generated image and the given text description using both global (sentence-level) and local (word-level) information.

**details on main method:**
Fig. 2 and Eq. 1 together gives a nice description of the main architecture. 
The model is a multi-scale GAN and attention is used in each scale.
The initial scale has no attention and is conditioned on the full text description.
Each following scale takes as input the previously generated image and the output from the attention module on the previous image and the word-level features.

The objective function is given in Eq. 3.
It is a combination of a regular GAN loss (with a twist that the discriminator also takes as input the sentence-level feature vector) and the so-called DAMSM loss.
This DAMSM loss is discussed in detail in Sec. 3.2.
The basic idea is that a bi-directional LSTM and a CNN first encode the text and the image, respectively.
Then based on these encoded vectors, the authors defined a similarity between an image and a text description (Eq. 10).
This similarity can be defined using encoded sentence or encoded individual words.
Given a batch of image-description pairs, the authors then defined the probability of a text description being matched with an image via softmax over the similarities (Eq. 11). 
The DAMSM loss amounts to minimizing the negative log probability of each text description being matched to the correct image, at both sentence-level and word-level. 

**additional comments:** 
- Results:
    - The DAMSM loss promotes semantic coherency with the given text (Fig. 3 and Table 2, shown by increased inception score and R-precision when the weight on the DAMSM term was increased). 
    - Intuitive ablation study results demonstrate clearly how the multi-scale architecture together with the attention mechanism helps learning fine-grained details from word-level information (Fig. 4).
    - Inception score comparisons against other GANs provided in Table 3, showing significant improvement on both datasets.



