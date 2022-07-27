
# AI/ML conference paper review 


## ICML 2022


### ✅  UQ，AD，OOD， Robustness

1. **Tackling covariate shift with node-based Bayesian neural networks** - [Oral] [PDF](https://proceedings.mlr.press/v162/trinh22a.html)

> Bayesian neural networks (BNNs) promise improved generalization under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weight-based BNNs often struggle with high computational complexity of large-scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent noise variables as implicit representations of simple and domain-agnostic data perturbations during training, producing BNNs that perform well under covariate shift due to input corruptions. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift due to input perturbations. As a side effect, the method also provides robustness against noisy training labels.

1. **How Tempering Fixes Data Augmentation in Bayesian Neural Networks** -[Oral] [PDF](https://proceedings.mlr.press/v162/bachmann22a.html)

> While Bayesian neural networks (BNNs) provide a sound and principled alternative to standard neural networks, an artificial sharpening of the posterior usually needs to be applied to reach comparable performance. This is in stark contrast to theory, dictating that given an adequate prior and a well-specified model, the untempered Bayesian posterior should achieve optimal performance. Despite the community’s extensive efforts, the observed gains in performance still remain disputed with several plausible causes pointing at its origin. While data augmentation has been empirically recognized as one of the main drivers of this effect, a theoretical account of its role, on the other hand, is largely missing. In this work we identify two interlaced factors concurrently influencing the strength of the cold posterior effect, namely the correlated nature of augmentations and the degree of invariance of the employed model to such transformations. By theoretically analyzing simplified settings, we prove that tempering implicitly reduces the misspecification arising from modeling augmentations as i.i.d. data. The temperature mimics the role of the effective sample size, reflecting the gain in information provided by the augmentations. We corroborate our theoretical findings with extensive empirical evaluations, scaling to realistic BNNs. By relying on the framework of group convolutions, we experiment with models of varying inherent degree of invariance, confirming its hypothesized relationship with the optimal temperature.


2. **Tractable Uncertainty for Structure Learning** - [Oral] [PDF](https://proceedings.mlr.press/v162/wang22ad.html) - [No Code]

> Bayesian structure learning allows one to capture uncertainty over the causal directed acyclic graph (DAG) responsible for generating given data. In this work, we present Tractable Uncertainty for STructure learning (TRUST), a framework for approximate posterior inference that relies on probabilistic circuits as a representation of our posterior belief. In contrast to sample-based posterior approximations, our representation can capture a much richer space of DAGs, while being able to tractably answer a range of useful inference queries. We empirically demonstrate how probabilistic circuits can be used to as an augmented representation for structure learning methods, leading to improvement in both the quality of inferred structures and posterior uncertainty. Experimental results also demonstrate the improved representational capacity of TRUST, outperforming competing methods on conditional query answering.

> 比较理论，没有code，但想法挺好，不再用samples to estimate uncertainty

3. **POEM: Out-of-Distribution Detection with Posterior Sampling** - [Oral] [PDF](https://proceedings.mlr.press/v162/ming22a.html)

> Out-of-distribution (OOD) detection is indispensable for machine learning models deployed in the open world. Recently, the use of an auxiliary outlier dataset during training (also known as outlier exposure) has shown promising performance. As the sample space for potential OOD data can be prohibitively large, sampling informative outliers is essential. In this work, we propose a novel posterior sampling based outlier mining framework, POEM, which facilitates efficient use of outlier data and promotes learning a compact decision boundary between ID and OOD data for improved detection. We show that POEM establishes state-of-the-art performance on common benchmarks. Compared to the current best method that uses a greedy sampling strategy, POEM improves the relative performance by 42.0% and 24.2% (FPR95) on CIFAR-10 and CIFAR-100, respectively. We further provide theoretical insights on the effectiveness of POEM for OOD detection.

> Sharon Li from WISC 的工作，很多OOD的工作还是挺不错的，关注的人很少， 有code！


4. **Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations** - [Oral] [PDF](https://proceedings.mlr.press/v162/zhang22z.html) [Code](https://github.com/HazyResearch/correct-n-contrast)

> Spurious correlations pose a major challenge for robust machine learning. Models trained with empirical risk minimization (ERM) may learn to rely on correlations between class labels and spurious attributes, leading to poor performance on data groups without these correlations. This is challenging to address when the spurious attribute labels are unavailable. To improve worst-group performance on spuriously correlated data without training attribute labels, we propose Correct-N-Contrast (CNC), a contrastive approach to directly learn representations robust to spurious correlations. As ERM models can be good spurious attribute predictors, CNC works by (1) using a trained ERM model’s outputs to identify samples with the same class but dissimilar spurious features, and (2) training a robust model with contrastive learning to learn similar representations for these samples. To support CNC, we introduce new connections between worst-group error and a representation alignment loss that CNC aims to minimize. We empirically observe that worst-group error closely tracks with alignment loss, and prove that the alignment loss over a class helps upper-bound the class’s worst-group vs. average error gap. On popular benchmarks, CNC reduces alignment loss drastically, and achieves state-of-the-art worst-group accuracy by 3.6% average absolute lift. CNC is also competitive with oracle methods that require group labels.

> 这个topic挺好的， paper的图画的不错，很多结果，大量的实验，不是很理论！Stanford group， 有code！


5. **BAMDT: Bayesian Additive Semi-Multivariate Decision Trees for Nonparametric Regression** -[Oral] [PDF](https://proceedings.mlr.press/v162/luo22a.html)

> Bayesian additive regression trees (BART; Chipman et al., 2010) have gained great popularity as a flexible nonparametric function estimation and modeling tool. Nearly all existing BART models rely on decision tree weak learners with axis-parallel univariate split rules to partition the Euclidean feature space into rectangular regions. In practice, however, many regression problems involve features with multivariate structures (e.g., spatial locations) possibly lying in a manifold, where rectangular partitions may fail to respect irregular intrinsic geometry and boundary constraints of the structured feature space. In this paper, we develop a new class of Bayesian additive multivariate decision tree models that combine univariate split rules for handling possibly high dimensional features without known multivariate structures and novel multivariate split rules for features with multivariate structures in each weak learner. The proposed multivariate split rules are built upon stochastic predictive spanning tree bipartition models on reference knots, which are capable of achieving highly flexible nonlinear decision boundaries on manifold feature spaces while enabling efficient dimension reduction computations. We demonstrate the superior performance of the proposed method using simulation data and a Sacramento housing price data set.

> 比较SOTA的regression，实验室用house pricing做的，如果做regression的话，可以参考


6. **Robustness Verification for Contrastive Learning** -[Oral] [PDF](https://proceedings.mlr.press/v162/wang22q.html) [Code](https://github.com/wzekai99/RVCL)

> Contrastive adversarial training has successfully improved the robustness of contrastive learning (CL). However, the robustness metric used in these methods is linked to attack algorithms, image labels and downstream tasks, all of which may affect the consistency and reliability of robustness metric for CL. To address these problems, this paper proposes a novel Robustness Verification framework for Contrastive Learning (RVCL). Furthermore, we use extreme value theory to reveal the relationship between the robust radius of the CL encoder and that of the supervised downstream task. Extensive experimental results on various benchmark models and datasets verify our theoretical findings, and further demonstrate that our proposed RVCL is able to evaluate the robustness of both models and images. Our code is available at https://github.com/wzekai99/RVCL.

> 有code，很多理论，不知道靠谱否，topic很有意思！

7. **A General Recipe for Likelihood-free Bayesian Optimization** - [Oral] [PDF](https://proceedings.mlr.press/v162/song22b.html) 

> The acquisition function, a critical component in Bayesian optimization (BO), can often be written as the expectation of a utility function under a surrogate model. However, to ensure that acquisition functions are tractable to optimize, restrictions must be placed on the surrogate model and utility function. To extend BO to a broader class of models and utilities, we propose likelihood-free BO (LFBO), an approach based on likelihood-free inference. LFBO directly models the acquisition function without having to separately perform inference with a probabilistic surrogate model. We show that computing the acquisition function in LFBO can be reduced to optimizing a weighted classification problem, which extends an existing likelihood-free density ratio estimation method related to probability of improvement (PI). By choosing the utility function for expected improvement (EI), LFBO outperforms the aforementioned method, as well as various state-of-the-art black-box optimization methods on several real-world optimization problems. LFBO can also leverage composite structures of the objective function, which further improves its regret by several orders of magnitude.

> 还是有人在做BO，这个实验是用image做的，Stanford group

8. **Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition** - [Oral] [PDF](https://proceedings.mlr.press/v162/wang22aq.html) 

> Existing out-of-distribution (OOD) detection methods are typically benchmarked on training sets with balanced class distributions. However, in real-world applications, it is common for the training sets to have long-tailed distributions. In this work, we first demonstrate that existing OOD detection methods commonly suffer from significant performance degradation when the training set is long-tail distributed. Through analysis, we posit that this is because the models struggle to distinguish the minority tail-class in-distribution samples, from the true OOD samples, making the tail classes more prone to be falsely detected as OOD. To solve this problem, we propose Partial and Asymmetric Supervised Contrastive Learning (PASCL), which explicitly encourages the model to distinguish between tail-class in-distribution samples and OOD samples. To further boost in-distribution classification accuracy, we propose Auxiliary Branch Finetuning, which uses two separate branches of BN and classification layers for anomaly detection and in-distribution classification, respectively. The intuition is that in-distribution and OOD anomaly data have different underlying distributions. Our method outperforms previous state-of-the-art method by 1.29, 1.45, 0.69 anomaly detection false positive rate (FPR) and 3.24, 4.06, 7.89 in-distribution classification accuracy on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT, respectively. Code and pre-trained models are available at https://github.com/amazon-research/long-tailed-ood-detection.


> Mu神组的工作， austin intern 


### ✅ Causal inference 

1.  **Causal Conceptions of Fairness and their Consequences** -[Oral] [PDF](https://proceedings.mlr.press/v162/nilforoshan22a.html) 

> Recent work highlights the role of causality in designing equitable decision-making algorithms. It is not immediately clear, however, how existing causal conceptions of fairness relate to one another, or what the consequences are of using these definitions as design principles. Here, we first assemble and categorize popular causal definitions of algorithmic fairness into two broad families: (1) those that constrain the effects of decisions on counterfactual disparities; and (2) those that constrain the effects of legally protected characteristics, like race and gender, on decisions. We then show, analytically and empirically, that both families of definitions almost always—in a measure theoretic sense—result in strongly Pareto dominated decision policies, meaning there is an alternative, unconstrained policy favored by every stakeholder with preferences drawn from a large, natural class. For example, in the case of college admissions decisions, policies constrained to satisfy causal fairness definitions would be disfavored by every stakeholder with neutral or positive preferences for both academic preparedness and diversity. Indeed, under a prominent definition of causal fairness, we prove the resulting policies require admitting all students with the same probability, regardless of academic qualifications or group membership. Our results highlight formal limitations and potential adverse consequences of common mathematical notions of causal fairness.

> 比较理论，但是causal 很多，这个paper是outstanding paper， 很solid，但是40 pages， stanford， NYU and harvord 




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


### ✅ AI for Science 

1. **Learning inverse folding from millions of predicted structures** - [Oral] [PDF](https://proceedings.mlr.press/v162/hsu22a/hsu22a.pdf)

> We consider the problem of predicting a protein sequence from its backbone atom coordinates. Machine learning approaches to this problem to date have been limited by the number of available experimentally determined protein structures. We augment training data by nearly three orders of magnitude by predicting structures for 12M protein sequences using AlphaFold2. Trained with this additional data, a sequence-to-sequence transformer with invariant geometric input processing layers achieves 51% native sequence recovery on structurally held-out backbones with 72% recovery for buried residues, an overall improvement of almost 10 percentage points over existing methods. The model generalizes to a variety of more complex tasks including design of protein complexes, partially masked structures, binding interfaces, and multiple states.

> 做alphafold的，然后是transformer架构 


2. **3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design** -[Oral] [PDF](https://proceedings.mlr.press/v162/huang22g.html) 

> Deep learning has achieved tremendous success in designing novel chemical compounds with desirable pharmaceutical properties. In this work, we focus on a new type of drug design problem — generating a small “linker” to physically attach two independent molecules with their distinct functions. The main computational challenges include: 1) the generation of linkers is conditional on the two given molecules, in contrast to generating complete molecules from scratch in previous works; 2) linkers heavily depend on the anchor atoms of the two molecules to be connected, which are not known beforehand; 3) 3D structures and orientations of the molecules need to be considered to avoid atom clashes, for which equivariance to E(3) group are necessary. To address these problems, we propose a conditional generative model, named 3DLinker, which is able to predict anchor atoms and jointly generate linker graphs and their 3D structures based on an E(3) equivariant graph variational autoencoder. So far as we know, no previous models could achieve this task. We compare our model with multiple conditional generative models modified from other molecular design tasks and find that our model has a significantly higher rate in recovering molecular graphs, and more importantly, accurately predicting the 3D coordinates of all the atoms.

3. **Generating 3D Molecules for Target Protein Binding** - [Oral] [PDF](https://proceedings.mlr.press/v162/liu22m.html)

> A fundamental problem in drug discovery is to design molecules that bind to specific proteins. To tackle this problem using machine learning methods, here we propose a novel and effective framework, known as GraphBP, to generate 3D molecules that bind to given proteins by placing atoms of specific types and locations to the given binding site one by one. In particular, at each step, we first employ a 3D graph neural network to obtain geometry-aware and chemically informative representations from the intermediate contextual information. Such context includes the given binding site and atoms placed in the previous steps. Second, to preserve the desirable equivariance property, we select a local reference atom according to the designed auxiliary classifiers and then construct a local spherical coordinate system. Finally, to place a new atom, we generate its atom type and relative location w.r.t. the constructed local coordinate system via a flow model. We also consider generating the variables of interest sequentially to capture the underlying dependencies among them. Experiments demonstrate that our GraphBP is effective to generate 3D molecules with binding ability to target protein binding sites. Our implementation is available at https://github.com/divelab/GraphBP.







<!-- # Robustness and Uncertainty in NLP
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
 -->
