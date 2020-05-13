---
title: "Modularizing Deep Learning via Pairwise Learning With Kernels"
collection: publications
permalink: /publication/duan2020modularizing
excerpt: '<b>TL;DR: Using a simple trick, we reveal the kernel machines hidden inside your favorite neural networks. Based on this observation, we propose a provably optimal modular training framework for neural networks in classification, making possible fully modular deep learning workflows. Our training method does not need between-module propagation and relies almost completely on weak pairwise labels yet still matches end-to-end backpropagation in accuracy. Finally, we demonstrate that a modular workflow naturally provides simple but reliable solutions to long-standing problems in important domains such as transfer learning.</b>'
venue: 'Preprint'
date: 2020-05-10
citation: '<b>Shiyu Duan</b>, Shujian Yu, Jose C. Principe, <i>preprint</i>'
paperurl: 'https://arxiv.org/pdf/2005.05541.pdf'
code: 'https://github.com/michaelshiyu/kerNET'
---
**TL;DR: Using a simple trick, we reveal the kernel machines hidden inside your favorite neural networks. Based on this observation, we propose a provably optimal modular training framework for neural networks in classification, making possible fully modular deep learning workflows. Our training method does not need between-module propagation and relies almost completely on weak pairwise labels yet still matches end-to-end backpropagation in accuracy. Finally, we demonstrate that a modular workflow naturally provides simple but reliable solutions to long-standing problems in important domains such as transfer learning.** 

&nbsp;

**Abstract**
    By redefining the conventional notions of layers, we present an alternative view on finitely wide, fully trainable deep neural networks as stacked linear models in feature spaces, leading to a kernel machine interpretation.
    Based on this construction, we then propose a provably optimal modular learning framework for classification, avoiding between-module backpropagation.
    This modular training approach brings new insights into the label requirement of deep learning:
    It leverages weak pairwise labels when learning the hidden modules.
    When training the output module, on the other hand, it requires full supervision but achieves high label efficiency, needing as few as 10 randomly selected labeled examples (one from each class) to achieve 94.88% accuracy on CIFAR-10 using a ResNet-18 backbone.
    Moreover, modular training enables fully modularized deep learning workflows, which then simplify the design and implementation of pipelines and improve the maintainability and reusability of models.
    To showcase the advantages of such a modularized workflow, we describe a simple yet reliable method for estimating reusability of pre-trained modules as well as task transferability in a transfer learning setting.
    At practically no computation overhead, it precisely described the task space structure of 15 binary classification tasks from CIFAR-10.

&nbsp;

**BibTeX**
```angular2
@misc{2005.05541,
Author = {Shiyu Duan and Shujian Yu and Jose Principe},
Title = {Modularizing Deep Learning via Pairwise Learning With Kernels},
Year = {2020},
Eprint = {arXiv:2005.05541},
}```
