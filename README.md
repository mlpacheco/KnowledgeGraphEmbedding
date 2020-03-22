**Adaptation**
This code was adapted to work for Argumentation Mining and Debate Stance
Prediction tasks to serve as baselines for my own research on deep
relational learning for NLP. The relational objectives where connected
to powerful language encoders and features relevant to the tasks at hand
(BiLSTMs, BERT, etc.)

# The code was adapted from: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
**Introduction**

This is the PyTorch implementation of the [RotatE](https://openreview.net/forum?id=HkgEQnRqYQ) model for knowledge graph embedding (KGE). We provide a toolkit that gives state-of-the-art performance of several popular KGE models. The toolkit is quite efficient, which is able to train a large KGE model within a few hours on a single GPU.

A faster multi-GPU implementation of RotatE and other KGE models is available in [GraphVite](https://github.com/DeepGraphLearning/graphvite).

**Implemented features**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult


**Citation**

Original implementation taken from the following [paper](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```

