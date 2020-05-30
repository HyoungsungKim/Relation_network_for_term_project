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

## Ensemble methods: bagging, boosting and stacking

https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

We will discuss some well known notions such as bootstrapping, bagging,  random forest, boosting, stacking and many others that are the basis of  ensemble learning.

### What are ensemble methods?

Ensemble learning is a machine learning paradigm where multiple models (often called “weak learners”) are trained to solve the same problem and combined to get better results.

#### Single weak learner

A low bias and a low variance are the two most fundamental features expected for a model. This is the well known **bias-variance tradeoff**.

In ensemble learning theory, we call **weak learners** (or **base models**) models that can be used as building blocks for designing more complex models by combining several of them.

- Most of the time, these basics models perform not so well by themselves either because they have a high bias (low degree of freedom models, for example) or because they have too much variance to be robust (high degree of freedom models, for example).
- Then, the idea of ensemble methods is to try reducing bias and/or variance of such weak learners by combining several of them together in order to create a **strong learner** (or **ensemble model**) that achieves better performances.

#### Combine Weak Learners

We first need to select our base models to be aggregated. 

- One important point is that our choice of weak learners should be **coherent with the way we aggregate these models**.
- If we ***choose base models with low bias but high variance***, it should be with an aggregating method that tends to ***reduce variance*** whereas if we ***choose base models with low variance but high bias***, it should be with an aggregating method that ***tends to reduce bias***.

Then how...?

- **bagging**, that often considers ***homogeneous weak learners***, learns them independently from each other in parallel and combines them following some kind of deterministic averaging process
- **boosting**, that often considers ***homogeneous weak learners***, learns them sequentially in a very adaptive way (a base model depends on the previous ones) and combines them following a deterministic strategy
- **stacking**, that often considers ***heterogeneous weak learners***, learns them in parallel and combines them by training a meta-model to output a prediction based on the different weak models predictions

> homogeneous : 동질
>
> heterogeneous : 이질

Very roughly, we can say that ***bagging will mainly focus at getting an ensemble model with less variance*** than its components whereas ***boosting and stacking will mainly try to produce strong models less biased*** than their components (even if variance can also be reduced).

### Bagging

#### Bootstrap aggregating - parallel methods

“Bagging” (standing for “bootstrap aggregating”) aims at producing an ensemble model that is **more robust** than the individual models composing it.

Under some assumptions, these samples have pretty **good statistical properties**: 

1. First approximation, they can be seen as being drawn both directly from the true underlying (and often unknown) data distribution and independently from each others.
   - So, ***they can be considered as representative and independent samples of the true data distribution (almost i.i.d. samples)***. The hypothesis that have to be verified to make this approximation valid are twofold. 
     1. First, ***the size N of the initial dataset should be large enough to capture most of the complexity of the underlying distribution*** so that sampling from the dataset is a good approximation of sampling from the real distribution (**representativity**).
     2. Second, ***the size N of the dataset should be large enough compared to the size B of the bootstrap samples*** so that samples are not too much correlated (**independence**).
   - Notice that in the following, we will sometimes make reference to these properties (representativity and independence) of bootstrap samples: the reader should always keep in mind that **this is only an approximation**.

In order to estimate the variance of such an estimator, we need to evaluate it on several independent samples drawn from the distribution of interest.

- In most of the cases, considering truly independent samples would require too much data compared to the amount really available.
- We can then use bootstrapping to generate several bootstrap samples that can be considered as being “almost-representative” and “almost-independent” (almost i.i.d. samples).
- These bootstrap samples will allow us to approximate the variance of the estimator, by evaluating its value for each of them.

#### Bagging

When training a model we obtain a function that takes an input, returns an output and that is defined with respect to the training dataset.

- Due to the theoretical variance of the training dataset, the fitted model is also subject to variability: **if another dataset had been observed, we would have obtained a different model**.

The idea of bagging is then simple: ***we want to fit several independent models and “average” their predictions in order to obtain a model with a lower variance***.

- However, we can’t, in practice, ***fit fully independent models because it would require too much data***.
- So, ***we rely on the good “approximate properties” of bootstrap samples*** (representativity and independence) to  fit models that are almost independent.