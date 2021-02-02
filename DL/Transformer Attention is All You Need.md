# Transformer : Attention is All You Need

Author: Kyungmin Lee
Link: https://arxiv.org/abs/1706.03762
Tags: NLP, transformer
Thesis: Transformer has a big potential.
리뷰여부: Yes

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.

## Abstract

---

- Encoder & Decoder structure is dominant sequence transduction model.
- We propose a new simple network architecture, **the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely**. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

## 1. Introduction

---

- RNN, LSTM, GRU have been n firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.
- Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature **precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples**
- Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, **allowing modeling of dependencies without regard to their distance in the input or output sequences**

## 2. Background

---

- The goal of reducing sequential computation also forms using convolutional neural networks as basic building block, **computing hidden representations in parallel** for all input and output positions → the number of operations required to relate signals from two arbitrary input or output positions **grows in the distance between positions ⇒ difficult to learn dependencies between distant positions**

기존 seq2seq모델은 인코더에서 디코더로 넘겨줄때 context vector를 통해서 넘겨주는데 그 고정된 크기의 벡터에 인코더의 모든 정보를 담으려다보니 정보 손실이 발생해요. 근데 transformer 모델은 그런거 없이 다 attention만으로 처리하니 인풋의 길이에 크게 제한받지 않는다는 의미 아닐까요

- Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence
- To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution

## 3. Model Architecture

---

- Encoder - Decoder Structure
- Here, the encoder maps an input sequence of symbol representations $(x_1,...x_n)$ to a sequence of continuous representations $z=(z_1,...,z_n)$. Given z, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time.
- The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

![Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled.png](Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled.png)

Figure 1

### 3.1 Encoder & Decoder Stacks

- **Encoder**
    - A stack of N = 6 identical layers. Each layer has two sub-layers.
    1. Multi-head self-attention mechanism
    2. Simple, position-wise fully connected feed-forward network.
    - a residual connection around each of the two sub-layers, followed by layer normalization

    ⇒ the output of each sub-layer is $LayerNorm(x+Sublayer(x))$

    - s, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model}=512$

- Decoder
    - A a stack of N = 6 identical layers
    - Decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack
    - residual connections around each of the sub-layers, followed by layer normalization
    - modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions

    ⇒ This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

### 3.2 Attention

- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

![Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%201.png](Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%201.png)

Figure 2

**3.2.1 Scaled Dot-Product Attention**

- The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$

1. Additive attention : computes the compatibility function using a feed-forward network with a **single hidden layer**
2. Dot-product Attention : can be implemented using **highly optimized matrix multiplication** code
- additive attention outperforms dot product attention without scaling for larger values of $d_k$. Suspected that for large values of $d_k$,  the dot products grow large in magnitude, **pushing the softmax function into regions where it has extremely small gradients**. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

**3.2.2 Multi-Head Attention**

- Beneficial to linearly project the queries, keys and values $h$ times with different, learned
linear projections to $d_k, d_k, d_v$ dimensions in **parallel.**
- **Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.**

$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O, \newline where\ head_i=Attention(QW_i^Q, KW^K_I, VW_i^V$

- Where the projections are parameter matrices...

 $W_i^Q\in\R^{d_{model}\times d_k},\ W_i^K\in\R^{d_{model}\times d_k},\ W_i^V\in\R^{d_{model}\times d_v},\ W^O\in\R^{d_{model}\times hd_v}$

- employ $h=8$ parallel attention layers, or heads. $d_k=d_v=d_{model}/h=64$

**3.2.3 Applications of Attention in our Model Transformer**

- "Encoder-decoder attention layer" : the queries come **from the previous decoder layer**,
and the memory keys and values come **from the output of the encoder**

⇒ **allows every position in the decoder to attend over all positions in the input sequence.**

- "self-attention layer of Encoder" : all keys, values and queries come from the same place

⇒ **Each position in the encoder can attend to all positions in the previous layer of the
encoder**

- "self-attention layer of Decoder" : allow each position in the decoder **to attend to
all positions in the decoder** up to and including that position.
    - need to **prevent leftward information flow in the decoder** to preserve the auto-regressive property ⇒ inside of scaled dot-product attention by **masking out (setting to −∞)** all values in the input of the softmax which correspond to illegal connections

### 3.3 Position-wise Feed-Forward Networks

- applied to each position separately and identically
- consists of two linear transformations with a ReLU activation in between

$FFN(x)=max(0,xW_1+b_1)W_2+b_2$

- While the linear transformations are the same across different positions, they use **different parameters from layer to layer.** (two convolutions with kernel size 1)

### 3.4 Embeddings & Softmax

- use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$
- use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities
- In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$

### 3.5 Positional Encoding

- No recurrence & no convolution : in order for the model **to make use of the order of the sequence**, we must inject some information about the relative or absolute position of the tokens in the sequence.
- The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed.
- we use sine and cosine functions of different frequencies: The wavelengths form a geometric progression from 2π to 10000 · 2π

$PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}}),\quad PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})$

- we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$

## 4. Why Self-Attention

---

- total computational complexity per layer.
- Another is the amount of computation that can be parallelized,

    ⇒ the minimum number of sequential operations required.

- path length between long-range dependencies in the network.
    - the length of the paths forward and backward signals have to traverse in the network is important . **The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies**
    - compare the maximum path length between any two input and output positions in networks composed of the different layer types.

    ![Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%202.png](Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%202.png)

    Table 1

- As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix

## 5. Training

---

### 5.1 Training Data & Batching

- Sentences were encoded according to the target language and datasets

### 5.2 Hardware & Schedule

- one machine with 8 NVIDIA P100 GPUs
- each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours

### 5.3 Optimizer

- Adam with $\beta_1=0.9,\quad \beta_2=0.98,\quad \epsilon=10^{-9}$

$learning \ rate=d_{model}^{-0.5}\cdot min(step\_num^{-0.5}, step\_num\cdot warmup\_steps^{-1.5})$

⇒ increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number(4000).

### 5.4 Regularization

- **Residual Dropout ($P_{drop}=0.1$)**
    - apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized
    - apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
- **Label Smoothing :** employed label smoothing of value $\epsilon_{ls}=0.1$
    - This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score

## 6. Results

---

### 6.1 Machine Translation

![Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%203.png](Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%203.png)

Table 2

### 6.2 Model Variations

![Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%204.png](Transformer%20Attention%20is%20All%20You%20Need%207b1871cd75d74bdcb79583e173de4bdd/Untitled%204.png)

- (A) : While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
- (B) : reducing the attention key size $d_k$ hurts model quality.

⇒ determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial

- (C) : bigger models are better
- (D) : dropout is very helpful in avoiding over-fitting
- (E) : replace our sinusoidal positional encoding with learned positional embeddings, but nearly identical

## 7. Conclusion

---

- For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.
- We achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

### Reference

[https://www.youtube.com/watch?v=AA621UofTUA&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=8](https://www.youtube.com/watch?v=AA621UofTUA&list=PLRx0vPvlEmdADpce8aoBhNnDaaHQN1Typ&index=8)