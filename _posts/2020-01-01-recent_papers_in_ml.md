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

Some of the papers were pretty new when I read them and the authors might have published newer versions afterwards.
Therefore, I include a unique version identifier in the title of each paper, indicating the version of the paper to which my comments correspond to.
The rationale behind not directly making the hyperlink point to a pdf file (which would solve the aforementioned versioning issue) is to allow the readers of this post to be aware of potential newer versions hosted on arXiv or somewhere else as well as other resources that are separate from the main paper (e.g., PMLR and NIPS Proceedings both put main text and supplementary materials in two separate places). 

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

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (version: arXiv v1)](https://arxiv.org/abs/2001.07685)
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

[A Simple Framework for Contrastive Learning of Visual Representations (version: arXiv v1)](https://arxiv.org/abs/2002.05709)
======

**keywords:** self-supervised learning, image classification 

**code:** 

**datasets:** self-supervised, semi-supervised evaluated on: ImageNet; transfer learning evaluated on: Food, CIFAR-10/100, Birdsnap, SUN397, Cars, Aircraft, VOC2007, DTD, Pets, Caltech-101, Flowers  

**one-sentence summary:** Based on contrastive learning, the authors proposed a simple self-supervised representation learning framework and demonstrated an instantiation with an off-the-shelf ResNet-50 backbone that achieved state-of-the-art performance on ImageNet (76.5% top-1 self-supervised, 85.8% top-5 fine-tuned with 1% of the labels). 

**details on main method:** Fig. 2 and Alg. 1 together gives a clear description. The particular contrastive loss used (NT-Xent) is in Eq. 1. Two augmented versions of the same image form a positive pair in the contrastive loss, whereas the negative pairs are simply pairs of augmented versions of distinct images. The contrastive loss is computed after a trainable projection head (discarded after training) that projects the representation into a lower-dimensional space. 

**additional comments:** 
- The network backbone is a ResNet-50. 
    And the activation of the final average pooling layer is used as the representation (mentioned in Sec. 2.1). 
- Evaluation methods used: 
    - Linear evaluation (train a supervised linear classifier on top of the learned representations). Main results in Table 6;
    - Fine-tuned with few labels (semi-supervised). Main results in Table 7; 
    - Transfer learning (froze or fine-tuned the self-supervised components). Main results in Table 8. 
- Nice literature review.
- Somewhat unusual section names: it seems that all sections starting from Sec. 3 are experiments and ablation study. 
- Extensive ablation study.
- The 1x, 2x, and 4x denote the width multiplier for the ResNet-50. This is described in the beginning of Sec. 6. 

&nbsp; 

[AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks (version: arXiv v1)](https://arxiv.org/abs/1711.10485)
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

[Subclass Distillation (version: arXiv v1)](https://arxiv.org/abs/2002.03936)
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

&nbsp; 

[Neural Kernels Without Tangents (version: arXiv v2)](https://arxiv.org/abs/2003.02237)
======

**keywords:** kernel method, deep learning theory 

**code:** [available](https://github.com/modestyachts/neural_kernels_code) 

**datasets:** CIFAR-10, UCI, MNIST, CIFAR-100, CIFAR-10.1 

**one-sentence summary:** Equated common operations used to form layers of neural networks to kernels, and studied the correlation between performance of CNNs (consisting of convolution, average pooling, and ReLU layers) and that of the corresponding (compositional) kernels. 

**details on main method:** 
- Theoretical framework:
    - Defined a "bag of features" to be simply a set of vectors in a Hilbert space (an example is a set of image patches).
    - Then defined three elementary mappings on these bags of features: concatenation (concatenating the vectors), downsampling (averaging the vectors), and embedding (a nonlinear mapping on the vectors).
    - These maps map into new Hilbert spaces, on which kernels can be induced from the inner products associated with the new Hilbert spaces (Sec. 3.1).  
    - Explicitly characterized kernels corresponding to four neural network operations: convolution, average pooling, ReLU, and an uncommon "normalized Gaussian kernel" (Sec. 3.2). 
    - The resulting kernels are shown to be expectations of inner products between outputs of random neural networks (Eq. 2).

**additional comments:** 
- Experimental settings:
    - Used ZCA whitening instead of mean subtraction plus standard deviation normalization.
    - The training of the kernel machines was described in the end of Sec. 4.2.
    
- Results:
    - The best CNN achieved 93%/96% accuracy on CIFAR-10 (with only flip augmentation/with additional augmentations including cutout and random crops).
    - The best kernel machine achieved 90% accuracy on CIFAR-10 (with only flip augmentation).
    
&nbsp;

[Deep Neural Networks as Gaussian Processes (version: ICLR 2018)](https://openreview.net/forum?id=B1EA-M-0Z)
======

**keywords:** kernel method, deep learning theory 

**code:** [available](https://github.com/brain-research/nngp) 

**datasets:** MNIST, CIFAR-10 

**one-sentence summary:** Extended the classic result that equated a one-hidden-layer MLP to a Gaussian process (GP) when the hidden layer width goes to infinity to arbitrary MLPs. 
And experimentally evaluated the performance of the so-called NNGPs, which are GPs using kernels induced by deep neural networks (NNs).

**details on main method:** 
- The classic one-hidden-layer result (reviewed in Sec. 2.2):
    - Consider a one-hidden-layer MLP and assume that the weights and biases of the layers are i.i.d. random variables (r.v.s).
    - Given the activations of the input layer, one may invoke the CLT and show that each activation in the output layer, being a sum over i.i.d. r.v.s, is itself a normal r.v..
    - Now, for any activation on the output layer, one may construct a Gaussian process (indexed over the input data) with zero mean (follows by construction of the distributions from which the weights and biases are sampled respectively) and covariance matrix given in Eq. 2 (dependant on the nonlinearity following the input layer). 

- The main theoretical result (Sec. 2.3):
    - The argument proceeds in an iterative fashion, taking the layer widths to infinity sequentially such that the layers are one by one (from the input layer to the output layer) governed by a GP.
    - Any single activation in any given layer being a GP (indexed over input data) follows a straightforward application of CLT when the activations of the previous layer are given.
    - The covariance matrix of this GP depends on the choice of nonlinearity following the previous layer and is given in Eq. 4 and simplified in Eq. 5.
    - The covariance matrix can be computed analytically for some nonlinearities.
    In particular, for ReLU, the kernel recovers the classic arccosine kernel. 
    When analytical computation is not possible, a numerical approach is given in Sec. 2.5.

- Practical implications:
    - Bayesian training of NNs using GP priors (Sec. 2.4): the predictive distribution for a test input given training data is characterized by a normal r.v. with mean and variance given in Eq. 8 and 9, respectively.
    - Uncertainty estimation for free (using the variance of the predictive distribution). 
    
**additional comments:** 
- Experimental settings:
    - The baseline NNs are MLPs with identical width at each hidden layer and were trained with Adam on MSE.
    The nonlinearities used include ReLU and Tanh.
    
- Main results are presented in Table 1. 
  NNGP performs on par with the somewhat weak MLP baselines with the highest accuracy being 55.66% on CIFAR-10 using 45k training data (53.13% for the corresponding MLP baseline).
  
&nbsp;

[Neural Tangent Kernel: Convergence and Generalization in Neural Networks (version: NeurIPS 2018)](http://papers.nips.cc/paper/8076-neural-tangent-kernel-convergence-and-generalization-in-neural-networks)
======

**keywords:** kernel method, deep learning theory 

**code:** 

**datasets:** synthetic data, MNIST 

**one-sentence summary:** Related the training dynamics and generalization of MLPs with the so-called neural tangent kernels (NTKs). 

**details on main method:** 
- A few key constructions/observations: 
    - Considering the input distribution to be the empirical distribution on a finite dataset, which reduces expectations to sample means (in the end of page 2).
    - Given a time-dependent function following kernel gradient descent, the evolution of the cost is related to the positive definiteness (defined near the end of page 3) of the kernel (in the beginning of page 4). 
    This helps preserve convexity of the functional cost.
    - Sec. 3.1: with linear realization functions, optimizing the cost with gradient descent is equivalent to kernel gradient descent with the tangent kernel.
    When the number of parameters in the network tends to infinity, this tangent kernel converges to a fixed kernel (using insight from the random Fourier features paper (Eq. 2 in that paper)) so the optimization is essentially kernel gradient descent with respect to a fixed limiting kernel.
    - Sec. 4: with arbitrary realization functions, the network function evolves along the kernel gradient with respect to the NTK.

- Key results:
    - Theorem 1 gives an explicit characterization of the limit of the NTK at initialization as the layer widths tend to infinity.
    - Theorem 2 extends Theorem 1 to any arbitrary time step and presents a differential equation that describes the dynamics of the network function during training in the infinite widths limit.
    - Sec. 5 provides a concrete example using least-squares regression, in which the functional cost and the trajectory of the network function are explicitly given after solving the above differential equation.
    In particular, the error vanishes faster along eigenspaces corresponding to larger eigenvalues, justifying early-stopping.
    - Used synthetic data to verify (1) the convergence of the NTK as layer widths go to infinity (Sec. 6.1) and (2) the network function follows a normal distribution for regression cost (Sec. 6.2).
    - Used MNIST to verify that in the regression case, the network function converges to the target at exponential rates along principal components (Sec. 6.3).

- The (somewhat nonstandard) MLP parameterization is given in Sec. 2.

**additional comments:** 
- Viewing the training cost that is usually considered a function with respect to the network parameters as a function of the network function is interesting.

&nbsp; 

[On Exact Computation with an Infinitely Wide Neural Net (version: NeurIPS 2019)](https://papers.nips.cc/paper/9025-on-exact-computation-with-an-infinitely-wide-neural-net)
======

**keywords:** kernel method, deep learning theory 

**code:** [available](https://github.com/ruosongwang/CNTK) 

**datasets:** CIFAR-10 

**one-sentence summary:** 
Detailed an efficient algorithm for computing convolutional neural tangent kernels (CNTKs) with ReLU activation. 
In the regression with squared error setting, improved the asymptotic results in the original NTK paper and other related works to non-asymptotic ones. 
The best CNTK achieved 77% accuracy on CIFAR-10, which is a new state-of-the-art (assuming not considering "methods that tune the kernel using training data or use a neural network to extract features and then applying a kernel method on top of them" (footnote in page 3)).

**details on main method:** 
- Lemma 3.1 shows that the derivative of the network output w.r.t. time is fully characterized by a time-varying kernel matrix. 
- Fully-connected networks with ReLU
    - Theorem 3.1 shows that for fully-connected networks, the earlier kernel matrix, at initialization of the network, converges to the NTK given in Eq. 9 as the layer widths tend to infinite and gives the rate of convergence (a non-asymptotic result).
    - Theorem 3.2 shows that after infinite training time with gradient descent, the fully-connected neural network's prediction on a normalized test example converges to that of the kernel regression model with NTK with large probability as the layer widths of the former tend to infinite. 
The speed of convergence is also given (a non-asymptotic result).
Formulation of the kernel regression model is given in the beginning of page 6.
        - Interpretation: kernel regression with the NTK kernel, a model that does not require iterative training, shares the same generalization characteristics as a wide fully-connected neural network after infinite training. 
- CNNs
    - CNTKs corresponding to a vanilla CNN and a CNN with global average pooling (GAP) were described and recursive formulae that compute them were given.  
        - The time complexity for CNN with GAP was given in the end of page 7 (super-quadratic in training set size). 

**additional comments:** 
- Main results are in Table 1.
- Reformulated many of the NTK paper results without relying on the functional analysis framework therein, which helps understanding things from a new perspective.
- The classification of related works that connect neural networks and kernel method into the studied networks being weakly-trained or fully-trained is neat.
- The main conclusion is that infinitely-wide networks are much worse than their finitely-wide counterparts.
