
# AI/ML conference paper review 


## ICML 2022


### ✅  UQ 

1. **Tackling covariate shift with node-based Bayesian neural networks** - [Oral] [PDF](https://proceedings.mlr.press/v162/trinh22a.html)

> Bayesian neural networks (BNNs) promise improved generalization under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weight-based BNNs often struggle with high computational complexity of large-scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent noise variables as implicit representations of simple and domain-agnostic data perturbations during training, producing BNNs that perform well under covariate shift due to input corruptions. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift due to input perturbations. As a side effect, the method also provides robustness against noisy training labels.


2. **Tractable Uncertainty for Structure Learning** - [Oral] [PDF](https://proceedings.mlr.press/v162/wang22ad.html) - [No Code]

> Bayesian structure learning allows one to capture uncertainty over the causal directed acyclic graph (DAG) responsible for generating given data. In this work, we present Tractable Uncertainty for STructure learning (TRUST), a framework for approximate posterior inference that relies on probabilistic circuits as a representation of our posterior belief. In contrast to sample-based posterior approximations, our representation can capture a much richer space of DAGs, while being able to tractably answer a range of useful inference queries. We empirically demonstrate how probabilistic circuits can be used to as an augmented representation for structure learning methods, leading to improvement in both the quality of inferred structures and posterior uncertainty. Experimental results also demonstrate the improved representational capacity of TRUST, outperforming competing methods on conditional query answering.

> 比较理论，没有code，但想法挺好，不再用samples to estimate uncertainty


### ✅ Generative models 

1. **Equivariant Diffusion for Molecule Generation in 3D**  - [Oral] [PDF](https://proceedings.mlr.press/v162/hoogeboom22a.html)

> This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and the efficiency at training time.


### ✅ Transfer learning 

1. **Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning** - [Oral] [PDF](https://proceedings.mlr.press/v162/evci22a/evci22a.pdf)

> Transfer-learning methods aim to improve performance in a data-scarce target domain using a model pretrained on a data-rich source domain. A cost-efficient strategy, linear probing, involves freezing the source model and training a new classification head for the target domain. This strategy is outperformed by a more costly but state-of-the-art method – fine-tuning all parameters of the source model to the target domain – possibly because fine-tuning allows the model to leverage useful information from intermediate layers which is otherwise discarded by the later previously trained layers. We explore the hypothesis that these intermediate layers might be directly exploited. We propose a method, Head-to-Toe probing (Head2Toe), that selects features from all layers of the source model to train a classification head for the target-domain. In evaluations on the Visual Task Adaptation Benchmark-1k, Head2Toe matches performance obtained with fine-tuning on average while reducing training and storage cost hundred folds or more, but critically, for out-of-distribution transfer, Head2Toe outperforms fine-tuning. Code used in our experiments can be found in supplementary materials.











# Robustness and Uncertainty in NLP
Robustness, uncertainty, safety and trustworthiness in deep learning, e.g., NLP, CV, multimodality


### NLP important papers 


| Read | Year | Name                                                         | Brief introduction                | Citation |   PDF link |
| ------ | ---- | ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------ |  ------- |
| ✅ | 2017 | Transformer | Attentaion is all you need   | NeurIPS 2017 | [PDF](https://github.com/jxzhangjhu/Robustness-and-Uncertainty-in-NLP/blob/main/pdf/1706.03762.pdf) |  



### NLP tutorial 




### UQ in Text classification 

1. (KDD 2021) **Uncertainty-Aware Reliable Text Classification** [PDF](https://github.com/jxzhangjhu/Robustness-and-Uncertainty-in-NLP/blob/main/pdf/KDD2021.pdf)
	> first apply evidential uncertainty (ENN) to text classification task; solve OOD detection tasks leveraging the OE method; three datasets (OE) are used as the benchmark; design adversarial examples


1. (AAAI 2020) **Uncertainty-aware deep classifiers using generative models** [PDF]


1. (NeurIPS 2019) **Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift** [PDF]


1. (ICML 2020 UDL workshop) **Predictive uncertainty for probabilistic novelty detection in text classification** [PDF]


1. (AAAI 2019) **Quantifying uncertainties in natural language processing tasks** [PDF] 

1. (COLING 2020) **Word-Level Uncertainty Estimation for Black-Box Text Classifiers using RNNs**

1. (ACL 2020) **Uncertainty-aware curriculum learning for neural machine translation**

1. (ACL 2022) **Uncertainty Estimation of Transformer Predictions for Misclassification Detection** 


### Active learning (MC dropout) 

1. (2017, Arxiv) **Deep active learning for named entity recognition**


1. (2018, Arxiv) **Deep bayesian active learning for natural language processing: Results of a large-scale empirical study**


### UQ method 


1. (NeurIPS 2018) **Evidential deep learning to quantify classification uncertainty** [PDF](https://github.com/jxzhangjhu/Robustness-and-Uncertainty-in-NLP/blob/main/pdf/NeurIPS-2018-evidential-deep-learning-to-quantify-classification-uncertainty-Paper.pdf)

