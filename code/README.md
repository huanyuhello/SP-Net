# SP-Net: Slowly Progressing Dynamic Inference Networks (SP-Net)

This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [SP-Net: Slowly Progressing Dynamic Inference Networks] presented at ECCV 2022. 

Designed by Huanyu Wang, Wenhu Zhang, Shihao Su, Hui Wang, Zhenwei Miao, Xin Zhan, and Xi Li.


The code is based on the [PyTorch example for training ResNet on Imagenet](https://github.com/pytorch/examples/tree/master/imagenet).


## Abstract
Dynamic inference networks improve computational efficiency by executing a subset of network components, i.e., executing path, conditioned on input sample. Prevalent methods typically assign routers to computational blocks so that a computational block can be skipped or executed. However, such inference mechanisms are prone to suffer instability in the optimization of dynamic inference networks. First, a dynamic inference network is more sensitive to its routers than its computational blocks. Second, the components executed by the network vary with samples, resulting in unstable feature evolution throughout the network. To alleviate the problems above, we propose SP-Nets to slow down the progress from two aspects. First, we design a dynamic auxiliary module to slow down the progress in routers from the perspective of historical information. Moreover, we regularize the feature evolution directions across the network to smoothen the feature extraction in the aspect of information flow. As a result, we conduct extensive experiments on three widely used benchmarks and show that our proposed SP-Nets achieve state-of-the-art performance in terms of efficiency and accuracy.

Keywords: Dynamic Inference, Slowly Progressing, Executing Path Regularization, Feature Evolution Regularization. 

## Framework
<p align="center">
<img src="overview.pdf" alt="regularization" width="100%">
</p>


## Contact
huanyuhello at zju dot edu dot cn ~~
