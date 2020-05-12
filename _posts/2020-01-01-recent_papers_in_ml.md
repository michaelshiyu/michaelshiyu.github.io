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

A running list of papers reviewed on this page:

- FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (version: arXiv v1) 
- A Simple Framework for Contrastive Learning of Visual Representations (version: arXiv v1)
- AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks (version: arXiv v1)
- Subclass Distillation (version: arXiv v1)
- Neural Kernels Without Tangents (version: arXiv v2)
- Deep Neural Networks as Gaussian Processes (version: ICLR 2018)
- Neural Tangent Kernel: Convergence and Generalization in Neural Networks (version: NeurIPS 2018)
- On Exact Computation with an Infinitely Wide Neural Net (version: NeurIPS 2019)
- A Theoretical Analysis of Contrastive Unsupervised Representation Learning (version: ICML 2019)
- Putting an End to End-to-End: Gradient-Isolated Learning of Representations (version: NeurIPS 2019)
- Taskonomy: Disentangling Task Transfer Learning (version: CVPR 2018)
- Similarity of Neural Network Representations Revisited (version: ICML 2019)
- Task2Vec: Task Embedding for Meta-Learning (version: ICCV 2019)
- Transferability and Hardness of Supervised Classification Tasks (version: ICCV 2019)
- LEEP: A New Measure to Evaluate Transferability of Learned Representations (version: arXiv v1)

<!---
&nbsp; 

[Template]()
======

**keywords:** 

**code:** 

**datasets:** 

**one-sentence summary:** 

**more details:** 

**additional comments:** 
-->

&nbsp; 

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (version: arXiv v1)](https://arxiv.org/abs/2001.07685)
======

**keywords:** semi-supervised learning, image classification

**code:** [available](https://github.com/google-research/fixmatch)

**datasets:** CIFAR-10/100, SVHN, STL-10, ImageNet

**one-sentence summary:** Combines two existing semi-supervised learning techniques, consistency regularization and pseudo-labeling, in a simplistic way but achieves state-of-the-art results (in particular, 94.93%/88.61% on CIFAR-10 with 250/40 labels). 

**more details:** Is essentially a loss function of two terms: cross-entropy on weakly augmented labeled examples (Eq. 3) and cross-entropy on strongly augmented, unlabeled examples using (only confident enough) model predictions (on weakly augmented versions of these images) as artificial labels (Eq. 4). 

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

**more details:** Fig. 2 and Alg. 1 together gives a clear description. The particular contrastive loss used (NT-Xent) is in Eq. 1. Two augmented versions of the same image form a positive pair in the contrastive loss, whereas the negative pairs are simply pairs of augmented versions of distinct images. The contrastive loss is computed after a trainable projection head (discarded after training) that projects the representation into a lower-dimensional space. 

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

**more details:**
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

**more details:** 
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

**more details:** 
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

**more details:** 
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

**more details:** 
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

**more details:** 
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

&nbsp; 

[A Theoretical Analysis of Contrastive Unsupervised Representation Learning (version: ICML 2019)](http://proceedings.mlr.press/v97/saunshi19a.html)
======

**keywords:** unsupervised learning, contrastive learning 

**code:** 

**datasets:** CIFAR-100, Wiki-3029 (language, created for this paper), IMDb 

**one-sentence summary:** 
Studied the relationship between unsupervised contrastive learning and the downstream supervised task.
The main results are bounds on the supervised loss of the unsupervised contrastive loss minimizer. 

**more details:** 
- The main construction is presented in Sec. 2. Some key components include:
    - The joint distribution of a positive pair and the distribution of a negative example are defined in Eq. 1 and 2, resp.
    Class identities are assumed, but the class labels are integrated out in these definitions, making the set-up essentially unsupervised.
    For the supervised downstream task, the joint distribution of the example and its label is given in Eq. 3. 
    - The unsupervised representation learner is trained with the contrastive learning framework and is attached to a linear layer for the supervised task.
    - Considers the hinge loss and the logistic loss.
    These loss functions are ingeniously rewritten such that the unsupervised contrastive learning task and the downstream supervised task share the same loss function form. 
    The loss functions are formally defined in Eq. 4, 5, and 6.

- The main bound using a single negative example is given in Th. 4.5, where the supervised loss of the contrastive loss minimizer is bounded in terms of the unsupervised loss of any function in the function class considered plus what can be interpreted as the expressiveness of this function.
Under additional assumptions on the bounding function, the right hand side of this bound can be refined to essentially the supervised loss of the bounding function plus its expressiveness. 

- The main bound using multiple negative examples is given in Th. 6.1.

- Proposed a more general contrastive learning method where multiple positive and multiple negative examples are allowed.
In Prop. 6.2, this learning method is shown to give a tighter bound than the original one.

- Experimentally verified that using larger block sizes improves performance of unsupervised contrastive learning, which is an example of how "fresh insights derived from our framework can lead to improvements upon state-of-the-art models in this active area".

**additional comments:** 
- The main construction is very interesting in 
    - how it assumes class identities of data examples (necessary for the downstream task) but "hides away" the labels when dealing with the unsupervised contrastive learning task.
    - how it uses a single loss function form to encompass both the unsupervised and supervised tasks.
    
&nbsp; 

[Putting an End to End-to-End: Gradient-Isolated Learning of Representations (version: NeurIPS 2019)](https://papers.nips.cc/paper/8568-putting-an-end-to-end-to-end-gradient-isolated-learning-of-representations)
======

**keywords:** contrastive learning, modular deep learning 

**code:** 

**datasets:** STL-10 (provides an additional unlabeled trainset), LibriSpeech

**one-sentence summary:**
An extension of contrastive predictive coding (CPC) to modular training of deep architectures.
The hidden modules are trained with a variant of the InfoNCE loss (Eq. 1) without labels. 

**more details:** 
- Review CPC in Sec. 2:
    - The InfoNCE loss (Eq. 1) is an instantiation of the contrastive learning framework on sequential data.
    - An autoregressive model takes in a sequence of (possibly encoded) examples z_0, ..., z_t, and outputs a single example c_t, which constitutes a positive pair with the k-steps-ahead input z_{t + k}. The negative pairs are c_t and z_i's uniformly sampled from all input examples. 
    - A classifier and this autoregressive model are jointly trained to classify the positive pair from all pairs.
    - In the end, the autoregressive model learns to extract features that are consistent over neighboring patches but nonexistent between random patches.

- The main method, dubbed Greedy InfoMax (GIM), is a generalization of CPC and is illustrated in Fig. 1:
    - Each module is trained to optimize the InfoNCE loss without an autoregressive module.
    The pairs are simply the module outputs with the positive pair being the module output at step t and at step t + k (Eq. 3 and 4).
    - The InfoNCE classifiers are discarded afterwards.
    - To deal with long-term dependency in, e.g., speech recognition, an autoregressive model is added to the InfoNCE loss of some modules (Eq. 6).
    - Argued that the method can be interpreted as
        - Maximizing the mutual information between nearby patch representations, and
        - Maximizing the mutual information between representations in consecutive modules.
    - For images, to form a sequential input, each image is split into ordered patches (Sec. 4.1).
    - A linear classifier is trained with softmax + cross-entropy on top of the representations learned by these stacked encoding modules.

- The main advantages:
    - The ability to leverage unlabeled data.
    - Lower memory usage. 

- The main vision results are in Table 1.
The various methods compared therein are described in the "Results" subsection in the beginning of page 6.
Note that the GIM model in the table had all its three hidden modules trained in sync (end-to-end).
And training them purely greedily hurt performance (second to last paragraph of page 6).

- The main speech results are in Table 3.
The various methods compared therein are described in the "Results" subsection of page 7.

- Fig. 4 shows that each module improves upon its predecessors in terms of learning representations suitable for the downstream classification task.

**additional comments:** 

- Missing a justification for the optimality of this learning method.

&nbsp; 

[Taskonomy: Disentangling Task Transfer Learning (version: CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/html/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.html)
======

**keywords:** transfer learning 

**code:** [demo available](http://taskonomy.stanford.edu/) 

**datasets:** proposed an image dataset of 4 million indoor scenes that has annotations for every one of the tasks studied on every image 

**one-sentence summary:** 
Proposed a computational method that discovers task transferability and predicts the optimal transfer policy among a given set of vision (generalizable to other domains) tasks.
A practical implication is that one can leverage the obtained taxonomic map to reduce the need for labeled data when learning a target task.

**more details:** 

- Requires:
    - A user-specified set of tasks whose underlying structure is of interest.
    - A dataset "that has annotations for *every task on every image*".  
- The four main steps (illustrated in Fig. 2):
    - Train one fully supervised network for each source task.
    The networks are required to have compatible architectures to make the subsequent transfer possible (this work used autoencoders).
    - The encoders from the source networks are frozen.
    On each possible source-task combinations (both first-order, i.e., single source, and higher-order, i.e., multiple sources, are considered), a network head (called a transfer function) is trained to solve the target task.
    The performance of this new model quantifies the task transferability between the corresponding source(s) and target.
    The head is kept small and trained with a small amount of labeled data (details are in the end of Sec. 3.2 "Accessibility").
    The rationale is that if the source(s) are highly transferable for the target, the features extracted by the source network should be *easily* read out by the new head without extensive training or a very expressive head architecture.
    - Since different target tasks use loss functions that potentially have different scales and induce different learning dynamics, the authors proposed an ordinal approach to normalize the raw task transferability quantified by performance of the new network head.
    Essentially, instead of comparing raw loss values, the relative transferability between two sources is measured by counting how many times one source is better than the other on the test set.
    Then for each source, its overall transferability is measured using comparisons against all sources (specifically, this value is obtained by reading out the corresponding component of the principal eigenvector of a tournament ratio matrix). 
    Stack the transferability vector of each target to form the affinity matrix P.
    - Given a set of tasks, the corresponding affinity matrix, max transfer order, supervision budget, and relative importance of target tasks, the authors modeled the problem of maximizing joint performance of all tasks (sum of transfer performance values from the affinity matrix weighted by target task importance) as a constrained subgraph selection problems and solved it using boolean integer programming. 
    The algorithm outputs a binary vector x of length e + v with value 1 indicating that this particular task or transfer is included in the final optimal transfer graph, where e is the total number of tasks, and v being the total number of possible transfers.

- Performance evaluations:
    - The source networks used were well-trained, achieving comparable performance to state-of-the-arts models. 
    - Win rate is defined as the proportion of test data on which the learned transfer policy resulted in a model that is better than the baseline.
    The transfer functions were trained using a small subset of the validation set (1k to 16k images, Sec. 4 "Data Splits") and the transfer policy was compared against two baselines: a network trained from scratch using the same data as the transfer networks' (win rate against this baseline is dubbed "gain"), and a network trained with 120k images (win rate dubbed "quality", a stronger baseline). 
    - The main result is given in Fig. 9.
    The transfer policy outperformed the weak baseline (in terms of gain) and matched the strong baseline (quality) most of the times.
    - On novel target tasks that do not have source networks to be used for the transfer policy, transfer networks were trained with 16k images and the resulting optimal transfer policy has high gain but low quality in most cases (Fig. 10, left).
    Moreover, using the same amount of labeled data, it usually outperforms training from scratch, self-supervised methods (can be understood as user-specified, fixed transfer), and using fixed ImageNet features (Fig. 10, right). 
    - The authors tested that the performance improvements in terms of gain and quality are significant (Fig. 11).
    - The estimated task transferability using the proposed dataset generalizes to some extent to other datasets (Fig. 12).
    - The estimated task transferability is robust to many design choices (Sec. 5.2).

**additional comments:** 

- Neat literature review.

- Sec. 6 gives a nice discussion on the limitations of this study arising from some of the simplifying assumptions.

&nbsp; 

[Similarity of Neural Network Representations Revisited (version: ICML 2019)](http://proceedings.mlr.press/v97/kornblith19a.html)
======

**keywords:** representation learning 

**code:** [available](https://github.com/google-research/google-research/tree/master/representation_similarity) 

**datasets:** CIFAR-10, CIFAR-100 

**one-sentence summary:**
Studied "the design and analysis of a scalar similarity index" between matrices "that can be used to compare representations within and across neural networks, in order to help visualize and understand the effect of different factors of variation in deep learning". 

**more details:** 
- The similarity index operates on pairs of matrices that share the same number of rows (batch size) but not necessarily columns (layer width) (end of Sec. 1, "Problem Statement").

- Argued that an ideal similarity index should  
    - *Not* be invariant to invertible linear transformations since
        - such an index cannot distinguish between representations when layer width exceeds batch size (Theorem 1), and
        - in practice, neural network training is *not* invariant to arbitrary invertible linear transformations of inputs or activations. 
    - Be invariant to orthogonal transformations.
    - Be invariant to isotropic scaling, i.e., scalar multiplication.

- Reviewed several similarity measures and summarized their invariance properties (Table 1).

- On a pair of architecturally identical networks trained with different initializations, centered kernel alignment (CKA) (with both linear and Gaussian kernel) identified representations from architecturally corresponding layers as most similar more successfully than the other similarity measures studied (Fig. 2 & Table 2).

- Based on the previous observation, the authors claimed that CKA is more suitable for measuring representation similarity and proceeded to perform the following exploratory experiments:
    - CKA was used to determine when the marginal performance improvement in network depth diminishes in a vanilla CNN by measuring if any newly added layer produced representations similar to the previous layers (Fig. 3).
    In ResNets, CKA revealed no such pathology: the layers consistently produced novel representations. 
    - CKA was used to measure similarity between layers from architecturally distinct networks (Fig. 5 & 6). 
    Upper, wider layers tend to produce more similar representations.
    - CKA was used to show that models trained on different datasets yielded similar representations in upper layers.
    And these representations differ from those produced by untrained networks. 

**additional comments:** 
- The main argument on why CKA is preferred over other matrix similarity measures is that it outperformed other measures in the proposed sanity check: a similarity measure should be robust to random network initializations and assign high similarity to architecturally corresponding layers in two identical networks trained from different random initializations. 
However, it is very difficult, if possible at all, to verify that this intuition itself is true.

- The experiments outlined a series of interesting problems to be studied in deep learning such as "how deep is too deep".
A good representation similarity measure can be really helpful when it comes to providing quantitative answers to these questions. 

- The visualizations are nice.

- This work (Table 1 in particular) can be really handy for future works in this direction. 

&nbsp; 

[Task2Vec: Task Embedding for Meta-Learning (version: ICCV 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html)
======

**keywords:** meta learning, task similarity, transfer learning

**code:** 

**datasets:** fine-grained classification tasks including iNaturalist, CUB-200, iMaterialist, and DeepFashion  

**one-sentence summary:** 
Proposed a method to embed tasks into a vector space such that their similarity can be quantified as distance in that space.
The proposed embedding methods were demonstrated to be useful for selecting pre-trained models.

**more details:** 

- Intuition: Fisher information matrix (FIM) characterizes importance of specific weights to the classification task (beginning of Sec. 3).
    Therefore, when using a single "probe" network over all tasks of interest, the corresponding FIMs encode information that is relevant for solving each task in a way that can be compared across the tasks (beginning of Sec. 3.1).

- To obtain FIM, the authors used a feature extractor pre-trained on ImageNet and fine-tuned a classifier head on each task.
    The FIM is computed for the feature extractor parameters (beginning of Sec. 3.1).
    Two additional approximation steps were taken to ensure (1) computational overhead is manageable (2) the resulting task representation has fixed size (governed by the number of filters of the probe network) regardless of the task.
    - Refined estimation via using the solution to the L functional ("Robust Fisher computation").

- Defined the (a)symmetric TASK2VEC distance through the FIM (Sec. 3.3) to quantify task distance.

- MODEL2VEC aims at embedding specific models into the task space such that models closer to a task are likely to perform well in that task.
    The embedding is a sum of the task vector of the task this model was trained on and a learnable vector representing the particularities of this model.
    These particularity vectors are trained to predict the performance of a model on a query task given TASK2VEC distance between the task this model was trained on and this query task.
    - This requires an extra training phase that has O(N) time complexity, where N is the number of available experts.

- The authors demonstrated the usefulness of TASK2VEC and MODEL2VEC on a model selection problem, where pre-trained feature extractors were given and were to be selected for target tasks and a linear classifier was trained on each feature extractor to determine the its actual performance (transferability) on target task (Sec. 4.2).
    The models can be selected either based on the source tasks they were trained (using TASK2VEC) or using MODEL2VEC, the latter of which achieved better performance (Table 1).
    The best of the proposed model selection strategies outperformed random selection and always using a fixed feature extractor pre-trained on ImageNet, but still do not always identify the best model (Table 1).
    
- The proposed methods were shown to work fine in the data-scarce regime (Fig. 4) and with different network backbones (Table 2).

**additional comments:** 

- Pixel-labeling and regression tasks are discussed in the supplementary materials.

- +: large-scale experimental evaluation (1400+ total tasks).

&nbsp; 

[Transferability and Hardness of Supervised Classification Tasks (version: ICCV 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Transferability_and_Hardness_of_Supervised_Classification_Tasks_ICCV_2019_paper.html)
======

**keywords:** task similarity, transfer learning 

**code:** 

**datasets:** CelebA, Animals with Attributes 2, CUB-200  

**one-sentence summary:**
Proposed a task similarity estimation method for classification using the conditional entropy (CE) between two sequences of training labels defining the tasks (on the same input examples). 
The method does not depend on particular solutions trained to solve the tasks, contrasting existing works such as Taskonomy and Task2Vec.

**more details:** 
- The main factor differentiating this method from successful existing ones such as Taskonomy and Task2Vec is that this one does not extract task-characterizing information from a proxy trained model --- this is done via using the labeled data only and therefore may avoid any bias/particularity of proxy models. 

- The theoretical results rely on the assumption that (1) a task is characterized by a labeled dataset and the cross-entropy loss (2) the task pair of interest share the same input examples (3) there is an underlying classification model that assumes a two-module representation. 

- Defined true transferability between a source and a target task to be the best test accuracy from fine-tuning a classifier head on top of a frozen network body that minimizes the (training) cross-entropy of the source task (Def. 1).
    - Proposed to use the training log-likelihood of this hybrid network on the target task as a surrogate measure for true transferability if the network did not overfit (Eq. 5).
    - The main result (Th. 1) is actually on this surrogate instead of the true transferability.
    
- CE is defined in Def. 2 using the empirical distribution of the training labels.
    Conceptually, it estimates the randomness of one label given the other (or the amount of information we still do not know about one label given the other), assuming we treat the labels as random variables.
    
- The main result (Th. 1) lower bounds a surrogate of true transferability by the source task performance of the optimal source model minus the CE between the two label sequences.

- Transferability is estimated using the CE between the tasks. 
    Hardness is estimated using the transferability with respect to a trivial task.
    Specifically, hardness of a given task, defined as the negative log-likelihood of the optimal model (Eq. 13) is upper bounded by the CE between its label sequence with respect to a constant label sequence (Eq. 14). 

**additional comments:** 
- Results are intuitive and neatly presented. 

- CE is actually only a *part* of a *bound* on a *surrogate* of the true transferability. 
    The other part of the bound involves actually a trained model.
    Only using CE as estimations for transferability bypasses the need for pre-trained models but may make the estimations less accurate. 
    
&nbsp; 

[LEEP: A New Measure to Evaluate Transferability of Learned Representations (version: arXiv v1)](https://arxiv.org/abs/2002.12462)
======

**keywords:** task similarity, transfer learning 

**code:** 

**datasets:** CIFAR-100, FashionMNIST 

**one-sentence summary:**
An extension of "Transferability and Hardness of Supervised Classification Tasks" that removes the assumption that source and target tasks share the same input data by introducing a source network into the construction and let the source task be fully represented by this source network.
The proposed transferability measure, LEEP, is essentially the performance of the a hybrid network defined via attaching a dummy head on top of the source network that converts its output categorical distribution over the source task labels to a distribution over the target task labels.

**more details:** 
- Represents the source task with a pre-trained model and the target task with a labeled dataset.

- True transferability is defined as how well the pre-trained source model can solve the target task.
    Considered two transferred networks, whose test performance will quantify true transferability (Sec. 5.1): (1) a network head is trained on the frozen feature extractor from the source model to solve the target task (called the head re-training method), and (2) the new network head and the source feature extractor are fine-tuned together (the fine-tuning method). 
    
- The main transferability measure, LEEP, is defined as follows.
    - Let the target label be y and source label be z, and let the trained source model be f.
    f(x) is an approximation of the categorical distribution p(z|x).
    - A classifier for the target task should approximate p(y|x), this equals p(y, z|x) = p(y|z, x)p(z|x) with z integrated/summed out.
    - If we estimate the distribution of p(y|z, x) using the empirical distribution p'(y|z, x) (Step 2 in Sec. 2), we can obtain a target task classifier by multiplying p'(y|z, x) and f(x) and then summing out z.
    - LEEP is defined as the training set log-likelihood of this classifier (called expected empirical predictor (EEP)) on the target task.

- The main result is that LEEP lower bounds the optimal *training* log-likelihood obtained via training a classifier head on top of the frozen feature extractor from the source network f used in the definition of LEEP (Property 1).

- LEEP is related to the negative conditional entropy (NCE) as transferability measure proposed by the same authors in Property 2.
    This result requires that NCE uses the predicted labels from the source network as the true labels for the source task since in LEEP's set-up, the source task is completely represented by the source network and there are no true labels to use.
    
- Demonstrated that LEEP works well in small-data (Sec. 5.2) and imbalanced-class (Sec. 5.3) set-ups, with a meta-transfer learning method CNAPs (Sec. 5.4), with different source networks (Sec. 5.7), and can outperform H scores and NCE (Sec. 5.6).

- In the small-data regime, transferred models with higher LEEP scores sometimes converge faster and outperform models trained from scratch using target task data (Sec. 5.5). 


- Notable details in the experimental set-up:
    - The target tasks were obtained as random subsets of CIFAR-100 classes (Sec. 5.1).
    - Performance of the transferability measure is quantified via the Pearson correlation coefficient between the measure and the true transferability (same as in the NCE work).
    - The true transferability is estimated using the F1 score in the imbalanced classes set-up instead of raw accuracy of transferred models.

**additional comments:** 
- Table 1 summarizes most of the results and is really neat.
