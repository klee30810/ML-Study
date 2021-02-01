# Dropout : A Simple Way to Prevent Neural Networks from Overfitting

Link: https://jmlr.org/papers/v15/srivastava14a.html
Tags: dropout, overfitting
Thesis: Dropout brings good generalization accuracy by ensembling random neuron combinations.
리뷰여부: Yes
### Author : Kyungmin Lee

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), 1929-1958.

## Abstract

---

- DNN contains a number of parameters so that the likelihood of overfitting is large. Normal method to deal with overfitting is to combining predictions of many different large neural net at test time, whereas it takes to much time and budget.
- Dropout is to randomly drop units (along with their connections) from the neural network during training, thereby preventing units from co-adapting too much.
- This is better than other regularization methods

## Introduction

---

- DNN contains multiple non-linear hidden layers and this makes them very expressive models that learn very complicated I/O relationships, but large numbers of parameters have DNN suffer from overfitting. Normally, it is coped with L1 & L2 regularization and soft weight sharing (Nowlan and Hinton, 1992).
- The best way to regularize a fixed-sized model is to average the predictions of all possible setting of parameters, weighting each setting by its posterior probability given the training data, but it costs too much. Combining several models is most helpful when the individual models are different from each other. In this regard, dropping some connections makes a combination of NNs which have different architecture.
- Dropout prevents overfitting and provides a way of approximately combining exponentially many different neural network efficiently. The choice of which units to drop is random with a fixed probability p; however, the optimal probability of retention is usually close to 1.
- Dropout amounts to sampling a thinned network which consists of all the units survived through dropout process. Training a NN with dropout can be seen as training a collection of $2^n$ thinned networks with extensive weight sharing, where each thinned network gets trained very rarely.
- We found that training a network with dropout and using this approximate averaging method at test time leads to significantly lower generalization error in diverse fields. Graphical Models can also be applied.

## Motivation

---

- Sexual reproduction : taking half the gene of one parent and half the gene of the other, adding a very small amount of random mutation. It seems plausible that asexual reproduction should be a better way to optimize individual fitness because a good set of genes that have come to work well together can be passed on directly to the offspring.

## Model Description

---

$r_j^{(l)} \sim Bernoulli(p),\quad \widetilde{y}^{(l)}=r^{(l)}*y^{(l)} \newline \newline z_i^{(l+1)}=w_i^{(l+1)}\widetilde{y}^l+b_i^{(l+1)}, \quad y_i^{(l+1)}= f(z_i^{(l+1)})$

## Learning Dropout Nets - Training

---

### (1) Back Propagation

- SGD와 비슷. However, for each training case in a mini-batch, we sample a thinned network by dropping out units. Forward and backpropagation are done only on this thinned network. The gradients for each parameter are averaged over the training case in each mini-batch.
- Max-norm regularization - constraining the norm of the incoming weight vector at each hidden neuron unit to be upper bounded by a fixed constant c. $||w||_2 \leq c$.
- Using dropout along with max-norm regularization, large decaying learning rates and high momentum provides a significant boost.
- Constraining weight vectors to lie inside a ball of fixed radius makes it possible to use a huge learning rate without the possibility of weights blowing up. The noise provided by dropout then allows the optimization process to explore different regions of the weight space that would have been difficult to reach.

### (2) Unsupervised Pretraining

- Pretraining followed by finetuning with back propagation shows significant results.
- Initially concerned that the stochastic nature of dropout might wipe out the information in the pretrained weights, but the information in the pretrained weights seemed to be retained when the learning rates are chosen to be smaller.

## Experiments

---

- 거의 모든 field에서 great

## Conclusion

---

- Dropout is a technique for improving neural networks by reducing overfitting. Standard backpropagation leraning builds up brittle co-adaptations that work for the training data but **do not generalize to unseen data. Random dropout breaks up these co-adaptations by making the presence of any particular hidden unit unreliable.**
- One drawback is that it increases training time. A dropout network typically takes 2~3 times longer to train than a standard neural network for the same architecture. A major cause is that the parameters updates are very noisy. Each training case effectively tries to train a different random architecture. Therefore, the gradients that are being computed are not gradients of the final architecture. **It is likely that this stochasticity prevents overfitting.**
