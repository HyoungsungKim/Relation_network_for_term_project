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

Variance 줄임

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

First, we create multiple bootstrap samples so that each new bootstrap sample will act as another (almost) independent dataset drawn from true distribution.

- Then, we can **fit a weak learner for each of these samples and finally aggregate them such that we kind of “average” their outputs**
- And obtain an ensemble model with less variance that its components. Roughly speaking, as the bootstrap samples are approximatively independent and identically distributed (i.i.d.), so are the learned base models. Then, ***“averaging” weak learners outputs do not change the expected answer but reduce its variance*** (just like averaging i.i.d. random variables preserve expected value but reduce variance).

There are several possible ways to aggregate the multiple models fitted in parallel.

- For a regression problem, the outputs of individual models can literally be averaged to obtain the output of the ensemble model.
- For classification problem the class outputted by each model can be seen as a vote and the class that receives the majority of the votes is returned by the ensemble model (this is called **hard-voting**).
  - Still for a classification problem, we can also consider the probabilities of each classes returned by all the models, average these probabilities and keep the class with the highest average probability (this is called **soft-voting**).
  - Hard-voting : 투표해서 가장 많이 투표 받은 것 선택
  - Soft-voting : 모델의 결과의 확률분포를 평균내서 평균 확률분포에서 가장 높은 확률을 가진 것 선택
- Averages or votes can either be simple or weighted if any relevant weights can be used.

#### Random forests

Strong learners composed of multiple trees can be called “forests”.

- Trees that compose a forest can be chosen to be either shallow (few depths) or deep (lot of depths, if not fully grown).
  - ***Shallow trees have less variance but higher bias*** and then will be better choice for sequential methods that we will described thereafter.
  - ***Deep trees, on the other side, have low bias but high variance*** and are relevant choices for bagging method that is mainly focused at reducing variance.

The **random forest** approach is a bagging method where **deep trees**, fitted on bootstrap samples, are ***combined to produce an output with lower variance***.

- However, random forests also use another trick to make the multiple fitted trees a bit less correlated with each others:
  - When growing each tree, instead of only sampling over the observations in the dataset to generate a bootstrap sample, we also **sample over features and keep only a random subset of them to build the tree**.
- ***Sampling over features has indeed the effect that all trees do not look at the exact same information to make their decisions and, so, it reduces the correlation between the different returned outputs***.
- Another advantage of sampling over the features is that **it makes the decision making process more robust to missing data**:
  - observations (from the training dataset or not) with missing data can still be regressed or classified based on the trees that take into account only features where data are not missing.
- ***Thus, random forest algorithm combines the concepts of bagging and random feature subspace selection to create more robust models***.

### Boosting

Bias줄임( Variance도 줄이긴 함)

In **sequential methods** the different combined weak models are no longer fitted independently from each others.

- The idea is to fit models **iteratively** such that the training of model at a given step depends on the models fitted at the previous steps.
- “Boosting” is the most famous of these  approaches and it produces an ensemble model that is in general less  biased than the weak learners that compose it.

#### Boosting

Boosting methods work in the same spirit as bagging methods:

- we build a family of models that are aggregated to obtain a strong learner that performs better.
- However, unlike bagging that mainly aims at reducing variance, ***boosting is a technique that consists in fitting sequentially multiple weak learners*** in a very adaptive way:
  - each model in the sequence is fitted giving more importance to observations in the dataset that were badly handled by the previous models in the sequence.

Intuitively, each new model **focus its efforts on the most difficult observations** to fit up to now, so that we obtain, at the end of the process, a strong learner with lower bias (even if we can notice that boosting can also have the effect of reducing variance). ***Boosting, like bagging, can be used for regression as well as for classification problems***.

> Boosting은 bias, variance 다 줄이긴 하지만 주로 high bias - low variance 모델을 weak learner로 사용함
>
> - 일반적으로 이게 계산량이 적음

What information from previous models do we take into account when fitting current model?

How do we aggregate the current model to the previous ones?

- It will be described in Adaboost, Gradient boost

In a nutshell, ***these two meta-algorithms differ on how they create and aggregate the weak learners during the sequential process***.

- Adaptive boosting updates the weights attached to each of the training dataset observations
- Gradient boosting updates the value of these observations.

This main difference comes from the way both methods try to solve the optimization problem of finding the best model that can be written as a weighted sum of weak learners.

#### Adaptive boosting(Adaboost)

In adaptive boosting (often called “adaboost”), we try to define our ensemble model as a weighted sum of L weak learners
$$
s_L(\cdot) = \sum^L_{l=1}c_l \times w_l(\cdot)
$$

- $c_l$ : coefficient
- $w_l$ : Weak learner

Finding the best ensemble model with this form is a **difficult optimization problem**(Iterative approach is required).

- Then, instead of trying to solve it in one single shot (finding all the coefficients and weak learners that give the best overall additive model), we make use of an **iterative optimization process** that is much more tractable, even if it can lead to a sub-optimal solution.
- More especially, we add the weak learners one by one, looking at each iteration for the best possible pair (coefficient, weak learner) to add to the current ensemble model.

In other words, we define recurrently the $(s_l)$’s such that
$$
s_l(\cdot) = s_{l-1}(\cdot) + c_l \times w_l(\cdot) \\
(c_l(\cdot), w_l(\cdot)) = \underset{c, w(\cdot)}{argmin} \sum^N_{n=1}e(y_n, s_{l-1}(x_n) + c \times w(x_n))
$$

- $s_l$ is the model that fit the best the training data 
- Instead of optimizing “globally” over all the L models in the sum, we approximate the optimum by optimizing “locally” building and adding the weak learners to the strong model one by one.

More especially, when considering a binary classification, we can show that the adaboost algorithm can be re-written into a process that proceeds as follow.

- First, it **updates the observations weights** in the dataset and train a new weak learner with a special focus given  to the observations misclassified by the current ensemble model.
- Second, it **adds the weak learner to the weighted sum** according to an update coefficient that express the performances of this weak model: the better a weak learner performs, the more it  contributes to the strong learner.

#### Gradient boosting

***The main difference with adaptive boosting is in the definition of the sequential optimization process***.

- Indeed, gradient boosting **casts the problem into a gradient descent one**:
  - At each iteration we ***fit a weak learner to the opposite of the gradient*** of the current fitting error with respect to the current ensemble model.

Let’s try to clarify this last point. First, theoretical gradient descent process over the ensemble model can be written
$$
s_l(\cdot) = s_{l-1}(\cdot) - c_l \times \nabla_{s_{l-1}}E(s_{l-1})(\cdot)
$$
Assume that we want to use gradient boosting technique with a given family of weak models. At the very beginning of the algorithm (first model of the  sequence), the pseudo-residuals are set equal to the observation values. Then, we repeat L times (for the L models of the sequence) the following steps:

- Fit the best possible weak learner to pseudo-residuals (approximate the opposite of the gradient with respect to the current strong learner)
- Compute the value of the optimal step size that defines by how much we update the ensemble model in the direction of the new weak learner
- Update the ensemble model by adding the new weak learner multiplied by the step size (make a step of gradient descent) 
  - Step size가 learning rate랑 비슷한 역할 하는 듯?
- Compute new pseudo-residuals that indicate, for each observation, in which direction we would like to update next the ensemble model predictions

Notice that, while adaptive boosting tries to solve at each iteration exactly the “local” optimization problem (find the best weak learner and its coefficient to add to the strong model), gradient boosting uses instead a gradient descent approach and can more easily be adapted to large number of loss functions. Thus, **gradient boosting can be considered as a generalization of adaboost to arbitrary differentiable loss functions**.

### Stacking

Stacking mainly differ from bagging and boosting on two points.

- First stacking often considers heterogeneous weak learners (**different learning algorithms are combined**) whereas bagging and boosting consider mainly homogeneous weak learners.
- Second, stacking learns to combine the base models using a meta-model whereas bagging and boosting combine weak learners following deterministic algorithms.

#### Staking

The idea of stacking is to learn several different weak learners and **combine them by training a meta-model** to output predictions based on the multiple predictions returned by these weak models. 

- So, ***we need to define two things in order to build our stacking model***:
  - The L learners we want to fit
  - The meta-model that combines them.

For example, for a classification problem, we can choose as weak learners a KNN classifier, a logistic regression and a SVM, and decide to learn a neural network as meta-model. ***Then, the neural network will take as inputs the outputs of our three weak learners and will learn to return final predictions based on it***.

- Relation network few-shot learning 이랑 비슷하네...

We split the dataset in two folds because predictions on data that have been used for the training of the weak learners are **not relevant for the training of the meta-model**.

- Thus, an obvious ***drawback of this split of our dataset in two parts is that we only have half of the data to train the base models and half of the data to train the meta-model***.
- ***In order to overcome this limitation, we can however follow some kind of “k-fold cross-training” approach***  (similar to what is done in k-fold cross-validation) such that ***all the observations can be used to train the meta-model***:
  - For any observation, the prediction of the weak learners are done with instances of these weak learners trained on the k-1 folds that do not contain the considered observation.
    - In other words, it consists in training on k-1 fold in order to make predictions on the remaining fold and that  iteratively so that to obtain predictions for observations in any folds.
    - Doing so, we can produce relevant predictions for each observation of our dataset and then train our meta-model on all these predictions.