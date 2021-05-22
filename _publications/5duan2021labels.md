---
title: "Labels, Information, and Computation:
Efficient, Privacy-Preserving Learning Using Sufficient
Labels"
collection: publications
permalink: /publication/duan2021labels
excerpt: '<b>TL;DR: We propose training classifiers with a novel form of labeled data that is easier to obtain but is just as informative. This new form of labeled data, which we call sufficiently-labeled data, also naturally provides protection for user privacy.</b>'
venue: 'Preprint'
date: 2021-05-22
citation: '<b>Shiyu Duan</b>, Jose C. Principe, <i>preprint, 2021</i>'
paperurl: 'https://arxiv.org/abs/2104.09015'
---
**TL;DR: We propose training classifiers with a novel form of labeled data that is easier to obtain but is just as informative. This new form of labeled data, which we call sufficiently-labeled data, also naturally provides protection for user privacy.** 

&nbsp;

**Abstract**
    In supervised learning, obtaining a large set of fully-labeled training data is expensive.
    We show that we do not always need full label information on every single training example to train a competent classifier. 
    Specifically, inspired by the principle of sufficiency in statistics, we present a statistic (a summary) of the fully-labeled training set that captures almost all the relevant information for classification but at the same time is easier to obtain directly. 
    We call this statistic "sufficiently-labeled data" and prove its sufficiency and efficiency for finding the optimal hidden representations, on which competent classifier heads can be trained using as few as a single randomly-chosen fully-labeled example per class. 
    Sufficiently-labeled data can be obtained from annotators directly without collecting the fully-labeled data first. 
    And we prove that it is easier to directly obtain sufficiently-labeled data than obtaining fully-labeled data. 
    Furthermore, sufficiently-labeled data naturally preserves user privacy by storing relative, instead of absolute, information. 
    Extensive experimental results are provided to support our theory.
&nbsp;

**BibTeX**
```angular2
@article{duan2021labels,
  title={Labels, Information, and Computation: Efficient, Privacy-Preserving Learning Using Sufficient Labels},
  author={Duan, Shiyu and Principe, Jose C},
  journal={arXiv preprint arXiv:2104.09015},
  year={2021}
}
```
