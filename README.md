# AWESOME-EMOP
> Paper List for Expensive Multi-objective Optimization 

## Table of Contents
:monkey:[Related Frameworks](#related-frameworks)   
:monkey:[Related Benchmarks](#related-benchmarks)   
:monkey:[Related Datasets](#related-datasets)  
:monkey:[Related Reviews](#related-reviews)  
:monkey:[Related Conferences and Journals](#related-conferences-and-journals)   
:monkey:[Multi-objective Bayesian Optimization](#multiobjective-bayesian-optimization) 

## Related Frameworks
- [BoTorch](https://botorch.org/)
- [GPyTorch](https://gpytorch.ai/)
- [nevergrad](https://github.com/facebookresearch/nevergrad)
- [Geatpy](https://github.com/geatpy-dev/geatpy)
- [DEAP](https://github.com/DEAP/deap)
- [pymoo](https://pymoo.org/)
- [PlatEMO](https://github.com/BIMK/PlatEMO)
- [JMetal](https://github.com/jMetal/jMetal)
- [MOBOpt](https://github.com/ppgaluzio/MOBOpt)
- [Paver](https://github.com/coin-or/Paver)

## Related Benchmarks
- [mDTLZ](https://ieeexplore.ieee.org/document/8372962)
- [UF](https://ojs.aaai.org/index.php/AAAI/article/view/10664) 
- [WFG](https://ieeexplore.ieee.org/document/5353656) 
- [DTLZ](https://www.cs.bham.ac.uk/~jdk/parego/)
- [BBOB](https://numbbo.github.io/workshops/)
- [COCO](https://github.com/numbbo/coco)
- [Hyper-parameter Tuning](http://www2.imm.dtu.dk/pubdb/edoc/imm6284.pdf)\[[paper](http://www2.imm.dtu.dk/pubdb/edoc/imm6284.pdf), [code](https://github.com/rasmusbergpalm/DeepLearnToolbox)\]
- [RWCMOPs](https://www.sciencedirect.com/science/article/pii/S2210650221001231)
- [Real world Problems, RE](https://www.sciencedirect.com/science/article/pii/S1568494620300181), \[[Paper](https://ryojitanabe.github.io/reproblems/),[Code](https://github.com/happywhy/2021-RW-MOP)\]


## Related Reviews and Books
- (2016) [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- (2007) [Evolutionary Algorithms for Solving Multi-Objective Problems](https://link.springer.com/book/10.1007/978-0-387-36797-2) 
- (2006) [Gaussian Process for Machine Learning](https://gaussianprocess.org/gpml/) 

## Related Conferences and Journals
### Coferences
[AAAI](https://www.aaai.org/Library/AAAI/aaai-library.php), 
[IJCAI](https://www.ijcai.org/proceedings/2019/), 
[ICLR](https://openreview.net/group?id=ICLR.cc/2019/Conference), 
[ICML](https://icml.cc/Conferences/2018/Schedule), 
[NIPS](https://nips.cc/Conferences/2018/Schedule?type=Poster),
[UAI](https://www.auai.org/uai2023/),
[AISTATS](https://aistats.org/),
[PPSN](https://emo2023.liacs.leidenuniv.nl/category/ppsn/),
[GECCO](https://gecco-2023.sigevo.org/HomePage),
[CEC](https://2023.ieee-cec.org/)

### Journals
[IEEE Transactions On Evolutionary Computation](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4235),
[IEEE Transactions on Cybernetics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221036), 
[IEEE Transactions on Intelligent Transportation Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979),
[Knowledge-based System](https://www.sciencedirect.com/journal/knowledge-based-systems),
[Soft Computating](https://www.springer.com/journal/500),
[Journal of Global Optimization](https://www.springer.com/journal/10898)
[Evolutionary Computation](https://direct.mit.edu/evco),
[The Journal of Machine Learning Research](https://www.jmlr.org/),
[Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing),
[Journal of Optimization Theory and Applications](https://www.springer.com/journal/10957),
[Swarm and Evolutionary Computation](https://www.sciencedirect.com/journal/swarm-and-evolutionary-computation)

## Preliminaries of Single-objective Bayesian Optimization

- (2022) [Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces](https://openreview.net/pdf?id=e4Wf6112DI)\[[code](https://www.catalyzex.com/paper/arxiv:2210.02905/code)]
- (2021) [Risk-averse Heteroscedastic Bayesian Optimization](https://proceedings.neurips.cc/paper/2021/file/8f97d1d7e02158a83ceb2c14ff5372cd-Paper.pdf)\[[code](https://github.com/Avidereta/risk-averse-hetero-bo)\]
- (2021) [Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization](https://proceedings.neurips.cc/paper/2021/file/3fab5890d8113d0b5a4178201dc842ad-Paper.pdf)\[[code](https://github.com/pinghsieh/FSAF)\]

-（2021）[Bayesian Optimization with High-Dimensional Outputs](https://openreview.net/pdf?id=vDo__0UwFNo)\[[code](https://botorch.org/tutorials/composite_mtbo)\]
- (2021) [Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks](https://openreview.net/forum?id=wF-llA3k32)
- (2021) [BORE: Bayesian Optimization by Density-Ratio Estimation](https://icml.cc/virtual/2021/oral/10202)
- (2021) [Bayesian Optimization over Hybrid Spaces](https://icml.cc/virtual/2021/spotlight/9184)
- (2019) [Adaptive and safe bayesian optimization in high dimensions via one-dimensional subspaces](http://proceedings.mlr.press/v97/kirschner19a/kirschner19a.pdf)


## Multi-objective Bayesian Optimization
---
### Neural Information Processing Systems (NIPS)
- (2022) [Pareto Set Learning for Expensive Multi-Objective Optimization](https://openreview.net/forum?id=vriLTB2-O0G) \[[code](https://github.com/Xi-L/PSL-MOBO)\]
- (2022) [Joint Entropy Search for Multi-Objective Bayesian Optimization](https://openreview.net/forum?id=ZChgD8OoGds)\[[code](https://github.com/benmltu/JES)]
- (2022) [Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis](https://www.aclweb.org/anthology/2020.acl-main.401/) \[[code](http://www.iitp.ac.in/˜ai-nlp-ml/resources.html)\]
- (2021)[Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement](https://openreview.net/pdf?id=A7pvvrlv68)\[[code](https://botorch.org/)\]


### International Conference on Machine Learning (ICML)
- (2022) [Dual Low-Rank Multimodal Fusion](https://proceedings.mlr.press/v162/daulton22a.html)\[[code](https://github.com/facebookresearch/robust_mobo)\]


### International Conference on Learning Representations (ICLR)
- (2022) [Multi-objective optimization by learning space partitions](https://openreview.net/pdf?id=FlwzVjfMryn) 


### Association for the Advancement of Artificial Intelligence (AAAI)
- (2017) [Solving high-dimensional multi-objective optimization problems with low effective di- mensions](https://ojs.aaai.org/index.php/AAAI/article/view/10664)


### Association for Uncertainty in Artificial Intelligence (UAI)
- (2021) [High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://auai.org/uai2021/pdf/uai2021.207.pdf)\[[code](https://github.com/martinjankowiak/saasbo)\]



### International Joint Conferences on Artificial Intelligence (IJCAI)
- (2019) [Success Prediction on Crowdfunding with Multimodal Deep Learning](https://www.ijcai.org/proceedings/2019/0299.pdf)


### IEEE Transactions on Cybernetics
- (2023) [Choose Appropriate Subproblems for Collaborative Modeling in Expensive Multiobjective Optimization(https://ieeexplore.ieee.org/document/9626546)]\[[code](https://github.com/ZhenkunWang/MOEAD-ASS)\] 


### IEEE Transactions On Evolutionary Computation (TEVC)
- (2021) [Expensive Multi-Objective Evolutionary Optimization Assisted by Dominance Prediction](https://ieeexplore.ieee.org/document/9490636)
- (2016) [A surrogate-assisted reference vector guided evolutionary algorithm for computationally expensive many-objective optimization](https://ieeexplore.ieee.org/document/7723883)
- (2010) [Expensive multi-objective optimization by MOEA/D with Gaussian process model](https://ieeexplore.ieee.org/document/5353656)
- (2006) [ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multi-objective optimization problems](https://ieeexplore.ieee.org/document/1583627)






