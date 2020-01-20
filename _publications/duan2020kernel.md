---
title: "On Kernel Method-Based Connectionist Models and Supervised Deep Learning Without Backpropagation"
collection: publications
permalink: /publication/duan2020kernel
excerpt: '<b>TL;DR:</b> TODO'
venue: 'Neural Computation'
date: 2020-01-01
citation: '<b>Shiyu Duan</b>, Shujian Yu, Yunmei Chen, Jose C. Principe, <i>Neural computation, 2020</i>'
paperurl: 'http://michaelshiyu.github.io/files/duan2020kernel.pdf'
code: 'https://github.com/michaelshiyu/kerNET'
---
**TL;DR: TODO** 

&nbsp;

**Abstract**

We propose a novel family of connectionist models based on kernel machines and consider the problem of learning layer by layer a compositional hypothesis class (i.e., a feedforward, multilayer architecture) in a supervised setting. 

In terms of the models, we present a principled method to “kernelize” (partly or completely) any neural network (NN).
With this method, we obtain a counterpart of any given NN that is powered by kernel machines instead of neurons. 

In terms of learning, when learning a feedforward deep architecture in a supervised setting, one needs to train all the components simultaneously using backpropagation (BP) since there are no explicit targets for the hidden layers (Rumelhart, Hinton, & Williams, 1986). 
We consider without loss of generality the two-layer case and present a general framework that explicitly characterizes a target for the hidden layer that is optimal for minimizing the objective function of the network. 
This characterization then makes possible a purely greedy training scheme that learns one layer at a time, starting from the input layer. 
We provide instantiations of the abstract framework under certain architectures and objective functions. 
Based on these instantiations, we present a layer-wise training algorithm for an l-layer feedforward network for classification, where l ≥ 2 can be arbitrary.
This algorithm can be given an intuitive geometric interpretation that makes the learning dynamics transparent. 

Empirical results are provided to complement our theory. 
We show that the kernelized networks, trained layer-wise, compare favorably with classical kernel machines as well as other connectionist models trained by BP. 
We also visualize the inner workings of the greedy kernelized models to validate our claim on the transparency of the layer-wise algorithm.

&nbsp;

**BibTeX**
```angular2
@article{duan2020kernel,
  title={On Kernel Method--Based Connectionist Models and Supervised Deep Learning Without Backpropagation},
  author={Duan, Shiyu and Yu, Shujian and Chen, Yunmei and Principe, Jose C},
  journal={Neural computation},
  volume={32},
  number={1},
  pages={97--135},
  year={2020},
  publisher={MIT Press}
}
```
