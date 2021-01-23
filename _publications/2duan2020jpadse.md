---
title: "JPAD-SE: High-Level Semantics for Joint Perception-Accuracy-Distortion Enhancement in Image Compression"
collection: publications
permalink: /publication/duan2020jpadse
excerpt: '<b>TL;DR: We propose a generic GAN-based framework that enables existing image compression codecs to leverage high-level semantics. We then show that thanks to the use of semantics, these "semantically-enhanced" codecs produce more visually pleasing results, enable downstream machine learning algorithm to perform significantly better, and achieve favorable rate-distortion performance when compared to the originals.</b>'
venue: 'Preprint'
date: 2020-05-26
citation: '<b>Shiyu Duan</b>, Huaijin Chen, Jinwei Gu, <i>preprint, 2020</i>'
paperurl: 'https://arxiv.org/abs/2005.12810'
---
**TL;DR: We propose a generic GAN-based framework that enables existing image compression codecs to leverage high-level semantics. We then show that thanks to the use of semantics, these "semantically-enhanced" codecs produce more visually pleasing results, enable downstream machine learning algorithm to perform significantly better, and achieve favorable rate-distortion performance when compared to the originals.** 

&nbsp;

**Abstract**
    While humans can effortlessly transform complex visual scenes into simple words and the other way around by leveraging their high-level understanding of the content, conventional or the more recent learned image compression codecs do not seem to utilize the semantic meanings of visual content to its full potential. 
    Moreover, they focus mostly on rate-distortion and tend to underperform in perception quality especially in low bitrate regime, and often disregard the performance of downstream computer vision algorithms, which is a fast-growing consumer group of compressed images in addition to human viewers. 
    In this paper, we (1) present a generic framework that can enable any image codec to leverage high-level semantics, and (2) study the joint optimization of perception quality, accuracy of downstream computer vision task, and distortion. 
    Our idea is that given any codec, we utilize high-level semantics to augment the low-level visual features extracted by it and produce essentially a new, semantic-aware codec. 
    And we argue that semantic enhancement implicitly optimizes rate-perception-accuracy-distortion (R-PAD) performance. 
    To validate our claim, we perform extensive empirical evaluations and provide both quantitative and qualitative results.

&nbsp;

**BibTeX**
```angular2
@misc{2005.12810,
Author = {Shiyu Duan and Huaijin Chen and Jinwei Gu},
Title = {JPAD-SE: High-Level Semantics for Joint Perception-Accuracy-Distortion Enhancement in Image Compression},
Year = {2020},
Eprint = {arXiv:2005.12810},
}
```
