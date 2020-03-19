# DGTL-Net
Deep Generative Transfer Learning Network for Zero-Fault Diagnostics

## Objectives

Intelligent fault diagnosis becomes significantly important to guarantee reliability of current IT and industrial systems. Most intelligent diagnostic methods are commonly based on assumptions that data from different monitored objects are subject to the same distribution and there are sufficient
faulty samples for training the models. However, in reality, there are types of objects from different manufacturers or working in different working conditions. It results in distribution discrepancy and influences the generalization of intelligent diagnostic methods. Moreover, systems usually work in healthy state that faulty events rarely happen on most of them, or especially never occur on new ones. Thus, this project aims at developing a deep generative transfer learning network (DGTL-Net) for zero-fault diagnostics. The objectives can be summarized as:

1. To improve the generalization of transferring learning model when faulty samples are absent on target domain, a new deep learning network (DGTL-Net) is developed. The network consists of two sub-networks with shared feature extractor: one sub-network is a deep generative network which could generate fake faulty samples for new hard disks based on their healthy samples; the other is a deep transfer network that minimizes the distribution discrepancy between domains.
2. To guarantee the most optimal parameters of deep generative network and deep transfer network simultaneously, an iterative end-end training strategy is proposed for DGTL-Net. The entire training process is based on Expectation-Maximization (EM) methodology which is divided into two parts. 

![](.\pics\1.jpg)