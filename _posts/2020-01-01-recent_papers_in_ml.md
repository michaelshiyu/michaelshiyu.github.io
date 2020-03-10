---
title: 'Recent Papers in Machine Learning'
date: 2020-01-01
permalink: /posts/2020/01/blog-post-1/
tags:
  - research
---

I decided to type up the notes I make when I read papers as a reference for myself when I come back to these papers at some point in the future.
The papers here are some recent ones in machine learning.
They don't necessarily have a common topic or anything and I read them simply because I think they are pretty damn cool.

In the following, **one-sentence summary** sometimes means **(I tried to keep the summary short and ideally) one-sentence (but I failed and produced a multi-sentence) summary (instead)**.

<!---
&nbsp; 

[Template]()
======

**keywords:** 

**code:** 

**datasets:** 

**one-sentence summary:** 

**details on main method:** 

**additional comments:** 
-->

&nbsp; 

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
======

**keywords:** semi-supervised learning, image classification

**code:** [available](https://github.com/google-research/fixmatch)

**datasets:** CIFAR-10/100, SVHN, STL-10, ImageNet

**one-sentence summary:** Combines two existing semi-supervised learning techniques, consistency regularization and pseudo-labeling, in a simplistic way but achieves state-of-the-art results (in particular, 94.93%/88.61% on CIFAR-10 with 250/40 labels). 

**details on main method:** Is essentially a loss function of two terms: cross-entropy on weakly augmented labeled examples (Eq. 3) and cross-entropy on strongly augmented, unlabeled examples using (only confident enough) model predictions (on weakly augmented versions of these images) as artificial labels (Eq. 4). 

**additional comments:** 
- Augmentation:
    - Weak augmentation is flip-and-shift. Discussed in the beginning of Sec. 2.3.
    - Tested with both RandAugment and CTAugment as strong augmentation. Discussed respectively in the last two paragraphs of Sec. 2.3.
- The authors emphasized the importance of factors that are usually considered "miscellaneous" and gave detailed descriptions on them. 
    To facilitate a fair comparison across methods, the authors reimplemented the baselines and used the same network backbone and training protocol for all baselines (mentioned in the beginning of Sec. 4.1).
    Some that are of particular importance are (mentioned in the second paragraph of Sec. 2.4. Analysis in Sec. 5): 
    - Regularization. Used weight decay regularization. 
    - Optimizer. Used SGD with momentum.
    - Learning rate schedule. Used a cosine decay strategy (Eq. 5).
    - Exponential moving average on model parameters.
- Some main hyperparameters are given in the beginning of Sec. 4. A full list of hyperparameters is provided in the supplementary material.
- Nice literature review. 
- Extensive ablation study.

&nbsp; 

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

&nbsp; 

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

&nbsp; 

[Subclass Distillation](https://arxiv.org/abs/2002.03936)
======

**keywords:** distillation (transfer learning)

**code:** 

**datasets:** CIFAR-10, CelebA, Criteo click prediction, MNIST 

**one-sentence summary:** 
Distillation does not work well in classification with a small number of total classes, this work encourages the teacher to produce a subclass assignment for each example before the actual class assignment. 
The student now learns to match both the subclass assignment and the class assignment. 
Since one can choose the number of subclasses arbitrarily, one may set it to be greater than that of the classes such that the student can be trained better with this extra information transferred from the teacher.  

**details on main method:** 
The method is illustrated in Fig. 2.
The main modification to the architecture is a substitution of the original logit layer with a subclass logit layer. 
The size of this layer (and hence the number of imaginary subclasses) can be freely determined.
The final class prediction is simply obtained by summing over the corresponding subclasses predictions for each class. 

The teacher loss is given in Eq. 8 and is a weighted sum of the usual cross-entropy and the proposed auxiliary loss.
Specifically, the auxiliary loss defined in Eq. 6 is to encourage the teacher to produce distinct subclass assignments for distinct inputs (even when they are from the same class) and thus reveal more information about its inner workings to the student. 
The student loss is given in Eq. 4 and is a weighted sum of the usual cross-entropy and a distill loss (Eq. 3) that matches the student's subclass predictions with the teacher's.

**additional comments:** 
- The method can perform unsupervised subclass classification with only binary supervision and, with the help from the auxiliary loss, surpasses fully-unsupervised state-of-the-art on CIFAR-10 in Sec. 4.1.1 (Table 1).
    - To obtain subclass assignment, argmax was taken on the subclass activations and the permutation with the highest accuracy was used.
- The analysis on the gain in the amount of information from using subclass distillation is interesting (the second-to-last paragraph in Sec. 4.1.1). 
- Main results are in Table 2 (CIFAR-10, subclass structure known) and Table 3 (CelebA, subclass structure unknown).
