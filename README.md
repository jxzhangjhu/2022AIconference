
# AI/ML conference paper review 


## ICML 2022


### ✅  UQ，AD，OOD， Robustness

1. **Tackling covariate shift with node-based Bayesian neural networks** - [Oral] [PDF](https://proceedings.mlr.press/v162/trinh22a.html)

> Bayesian neural networks (BNNs) promise improved generalization under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weight-based BNNs often struggle with high computational complexity of large-scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent noise variables as implicit representations of simple and domain-agnostic data perturbations during training, producing BNNs that perform well under covariate shift due to input corruptions. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift due to input perturbations. As a side effect, the method also provides robustness against noisy training labels.


2. **Tractable Uncertainty for Structure Learning** - [Oral] [PDF](https://proceedings.mlr.press/v162/wang22ad.html) - [No Code]

> Bayesian structure learning allows one to capture uncertainty over the causal directed acyclic graph (DAG) responsible for generating given data. In this work, we present Tractable Uncertainty for STructure learning (TRUST), a framework for approximate posterior inference that relies on probabilistic circuits as a representation of our posterior belief. In contrast to sample-based posterior approximations, our representation can capture a much richer space of DAGs, while being able to tractably answer a range of useful inference queries. We empirically demonstrate how probabilistic circuits can be used to as an augmented representation for structure learning methods, leading to improvement in both the quality of inferred structures and posterior uncertainty. Experimental results also demonstrate the improved representational capacity of TRUST, outperforming competing methods on conditional query answering.

> 比较理论，没有code，但想法挺好，不再用samples to estimate uncertainty

3. **POEM: Out-of-Distribution Detection with Posterior Sampling** - [Oral] [PDF](https://proceedings.mlr.press/v162/ming22a.html)

> Out-of-distribution (OOD) detection is indispensable for machine learning models deployed in the open world. Recently, the use of an auxiliary outlier dataset during training (also known as outlier exposure) has shown promising performance. As the sample space for potential OOD data can be prohibitively large, sampling informative outliers is essential. In this work, we propose a novel posterior sampling based outlier mining framework, POEM, which facilitates efficient use of outlier data and promotes learning a compact decision boundary between ID and OOD data for improved detection. We show that POEM establishes state-of-the-art performance on common benchmarks. Compared to the current best method that uses a greedy sampling strategy, POEM improves the relative performance by 42.0% and 24.2% (FPR95) on CIFAR-10 and CIFAR-100, respectively. We further provide theoretical insights on the effectiveness of POEM for OOD detection.

> Sharon Li from WISC 的工作，很多OOD的工作还是挺不错的，关注的人很少， 有code！


4. **Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations** - [Oral] [PDF](https://proceedings.mlr.press/v162/zhang22z.html) [Code](https://github.com/HazyResearch/correct-n-contrast)

> Spurious correlations pose a major challenge for robust machine learning. Models trained with empirical risk minimization (ERM) may learn to rely on correlations between class labels and spurious attributes, leading to poor performance on data groups without these correlations. This is challenging to address when the spurious attribute labels are unavailable. To improve worst-group performance on spuriously correlated data without training attribute labels, we propose Correct-N-Contrast (CNC), a contrastive approach to directly learn representations robust to spurious correlations. As ERM models can be good spurious attribute predictors, CNC works by (1) using a trained ERM model’s outputs to identify samples with the same class but dissimilar spurious features, and (2) training a robust model with contrastive learning to learn similar representations for these samples. To support CNC, we introduce new connections between worst-group error and a representation alignment loss that CNC aims to minimize. We empirically observe that worst-group error closely tracks with alignment loss, and prove that the alignment loss over a class helps upper-bound the class’s worst-group vs. average error gap. On popular benchmarks, CNC reduces alignment loss drastically, and achieves state-of-the-art worst-group accuracy by 3.6% average absolute lift. CNC is also competitive with oracle methods that require group labels.

> 这个topic挺好的， paper的图画的不错，很多结果，大量的实验，不是很理论！Stanford group， 有code！



### ✅ Generative models 

1. **Equivariant Diffusion for Molecule Generation in 3D**  - [Oral] [PDF](https://proceedings.mlr.press/v162/hoogeboom22a.html)

> This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and the efficiency at training time.

2. **Path-Gradient Estimators for Continuous Normalizing Flows**  -[Oral] [PDF](https://proceedings.mlr.press/v162/vaitl22a.html)

> Recent work has established a path-gradient estimator for simple variational Gaussian distributions and has argued that the path-gradient is particularly beneficial in the regime in which the variational distribution approaches the exact target distribution. In many applications, this regime can however not be reached by a simple Gaussian variational distribution. In this work, we overcome this crucial limitation by proposing a path-gradient estimator for the considerably more expressive variational family of continuous normalizing flows. We outline an efficient algorithm to calculate this estimator and establish its superior performance empirically.

> continuous flow + variational 需要check 有没有code， 不是很复杂



### ✅ Multimodality (Vision, Speech, Lanuager, Graph)

1. **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** - [Oral] [PDF](https://proceedings.mlr.press/v162/baevski22a.html)

> While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

> MetaAI的工作，大活，估计很多人会follow


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

