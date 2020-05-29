# Literature survey

## Boosting Few-Shot Visual Learning with Self-Supervision

### Introduce

We propose to add a self-supervised loss to the training loss that a few-shot model minimizes during its first learning  stage.

- More specifically, we propose to add a self-supervised loss to the training loss that a few-shot model minimizes  during its first learning stage.

The contributions of our work are:

1. We propose to weave self-supervision into the training objective of few-shot learning algorithms.
   - The goal is to boost the ability of the latter to adapt to novel classes with few training data.
2. We study the impact of the added self-supervised loss by performing exhaustive quantitative experiments on MiniImagenet, CIFAR-FS, and tiered-MiniImagenet few-shot datasets
   - In all of them self-supervision improves the few-shot learning performance leading to state-of-the-art results.
3. Finally, ***we extend the proposed few-shot recognition framework to semi-supervised and unsupervised setups***, getting further performance gain in the former, and showing with the latter that our framework can be used for evaluating and comparing self-supervised representation learning approaches on few-shot object recognition.

### Boosting few-shot learning via self-supervision

A major challenge in few-shot learning is encountered during the first stage of learning.

- How to make the feature extractor learn image features that can be readily exploited for novel classes with few training data during the second stage?

With this goal in mind, we propose to leverage the recent progress in self-supervised feature learning to further improve current few-shot learning approaches.

- We propose to extend the training of the feature extractor $F_\theta(.)$ by including such a self-supervised task besides the main task of recognizing base classes.

For the self-supervised loss, ***we consider two tasks in the present work***:

- Predicting the rotation incurred by an image, which is simple and readily incorporated into a few-shot learning algorithm;
- Predicting the relative location of two patches from the same image, a seminal task in self-supervised learning. 

 Few-shot learning은 2개의 stage로 나누어져 있는데 1번째 스테이지에서 feature를 어떻게 잘 뽑아내느냐가 정확도를 결정함. 두번째 스태이지는 metric 사용하는곳으로 metric은 cosine distance, euclidean distance, CNN 등등 

## Random Forest Classifier for Zero-Shot Learning Based on Relative Attribute

We therefore propose a novel zero-shot image classifier called random forest based on relative attribute. 

- First, based on the ordered and unordered pairs of images from the seen classes, the idea of ranking support  vector machine is used to learn ranking functions for attributes.
- Then, according to the relative relationship between seen and unseen classes, the RA(Relative Attributes) ranking-score model per attribute for each unseen image is built, where the appropriate seen classes are automatically selected to participate in the modeling process.
- In the third step, the random forest classifier is trained based on the RA ranking scores of attributes for all seen and  unseen images.
- Finally, the class labels of testing images can be predicted via the trained RF.

### Introduction

Different object classes may have common attributes. For example, horse and giraffe are both quadrupeds and share common topologies. If we adopt these attributes as a mid-layer of an object classifier to allow different objects to share  certain common attributes, ***the prior knowledge about attributes can be transferred from known(seen) classes to unknown(unseen) classes***.

- Such a classifier can address the object recognition problem when no training samples are available.

***The attribute-based zero-shot learning methods generally use low-level features of images to train attribute classifiers,  and the corresponding CA(Classification accuracy) heavily depends on specific low-level features***.

- In recent years, inspired by the observation that ***deep networks are capable of automatically extracting features*** from original images and the extracted features can better represent the nature of original images, respectively, proposed a model that integrates RA framework with deep convolutional neural networks to improve the CA.
- Li et al. employed RA as a base ranking function to construct a relative tree according to the maximal information gain criterion.

Due to the hierarchical tree structure, the proposed relative tree can efficiently capture complex nonlinear structure of feature manifold and generate a piecewise linear ranking function to rank the nonlinear data accurately.

In general, ensemble learning can improve generalization performance under various circumstances. ***Random forest is an ensemble classifier with a decision tree as base classifier***.

- The decision tree is applicable for attribute-based classification problems. The decision tree has advantages of being intuitive and is easy to interpret.
- In addition, it does not require that the underlying distribution is known in advance.
- Therefore, a random forest classifier based on RA (RF-RA) for zero-shot learning is proposed.

In this paper, our main contributions are outlined as the following two aspects:

1. We presented a modeling process of RA ranking score for each unseen image according to the relative relationship between seen and unseen classes, where the assumption of all seen and unseen images obeying a definite distribution (e.g., Gaussian distribution in traditional RA) is unnecessary to be satisfied
2. We proposed a novel zero-shot image classifier (RF-RA) to realize the non-linear mapping from RA ranking scores of testing images to class labels, overcoming the drawback of MLE