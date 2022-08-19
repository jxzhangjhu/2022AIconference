
# AI/ML conference paper review 


## Review paper

1. **Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons** [pdf](https://arxiv.org/pdf/2201.07766.pdf)

> George组的paper 比较全面，时间到2022.1月, 有时间看一下



## ICML & ICLR 2022


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

[
9. **Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation** -[Oral][PDF](https://proceedings.mlr.press/v162/shen22d.html)

> We consider unsupervised domain adaptation (UDA), where labeled data from a source domain (e.g., photos) and unlabeled data from a target domain (e.g., sketches) are used to learn a classifier for the target domain. Conventional UDA methods (e.g., domain adversarial training) learn domain-invariant features to generalize from the source domain to the target domain. In this paper, we show that contrastive pre-training, which learns features on unlabeled source and target data and then fine-tunes on labeled source data, is competitive with strong UDA methods. However, we find that contrastive pre-training does not learn domain-invariant features, diverging from conventional UDA intuitions. We show theoretically that contrastive pre-training can learn features that vary subtantially across domains but still generalize to the target domain, by disentangling domain and class information. We empirically validate our theory on benchmark vision datasets.

> Stanford group 做的，比较理论，实验比较弱


10. **Bayesian Model Selection, the Marginal Likelihood, and Generalization** - [Oral][PDF](https://proceedings.mlr.press/v162/lotfi22a/lotfi22a.pdf) 

> How do we compare between hypotheses that are entirely consistent with observations? The marginal likelihood (aka Bayesian evidence), which represents the probability of generating our observations from a prior, provides a distinctive approach to this foundational question, automatically encoding Occam's razor. Although it has been observed that the marginal likelihood can overfit and is sensitive to prior assumptions, its limitations for hyperparameter learning and discrete model comparison have not been thoroughly investigated. We first revisit the appealing properties of the marginal likelihood for learning constraints and hypothesis testing. We then highlight the conceptual and practical issues in using the marginal likelihood as a proxy for generalization. Namely, we show how marginal likelihood can be negatively correlated with generalization, with implications for neural architecture search, and can lead to both underfitting and overfitting in hyperparameter learning. We provide a partial remedy through a conditional marginal likelihood, which we show is more aligned with generalization, and practically valuable for large-scale hyperparameter learning, such as in deep kernel learning.

> outstanding paper 有争议，这个paper 有code? 


11. **Near-Exact Recovery for Tomographic Inverse Problems via Deep Learning** -[Oral][PDF](https://proceedings.mlr.press/v162/genzel22a.html)

> This work is concerned with the following fundamental question in scientific machine learning: Can deep-learning-based methods solve noise-free inverse problems to near-perfect accuracy? Positive evidence is provided for the first time, focusing on a prototypical computed tomography (CT) setup. We demonstrate that an iterative end-to-end network scheme enables reconstructions close to numerical precision, comparable to classical compressed sensing strategies. Our results build on our winning submission to the recent AAPM DL-Sparse-View CT Challenge. Its goal was to identify the state-of-the-art in solving the sparse-view CT inverse problem with data-driven techniques. A specific difficulty of the challenge setup was that the precise forward model remained unknown to the participants. Therefore, a key feature of our approach was to initially estimate the unknown fanbeam geometry in a data-driven calibration step. Apart from an in-depth analysis of our methodology, we also demonstrate its state-of-the-art performance on the open-access real-world dataset LoDoPaB CT.

> 不是特别典型的inverse paper，用的CT data 可以参考一下


12. **Certified Robustness Against Natural Language Attacks by Causal Intervention** [PDF](https://proceedings.mlr.press/v162/zhao22g.html)

> Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.8% in terms of certified robustness against word substitutions, and achieves 80.7% empirical robustness when syntactic attacks are integrated.

> NLP 相关的robustness + causal 还是比较热门的方向


13. **ButterflyFlow: Building Invertible Layers with Butterfly Matrices** -[PDF](https://proceedings.mlr.press/v162/meng22a.html)

> Normalizing flows model complex probability distributions using maps obtained by composing invertible layers. Special linear layers such as masked and 1{\texttimes}1 convolutions play a key role in existing architectures because they increase expressive power while having tractable Jacobians and inverses. We propose a new family of invertible linear layers based on butterfly layers, which are known to theoretically capture complex linear structures including permutations and periodicity, yet can be inverted efficiently. This representational power is a key advantage of our approach, as such structures are common in many real-world datasets. Based on our invertible butterfly layers, we construct a new class of normalizing flow mod- els called ButterflyFlow. Empirically, we demonstrate that ButterflyFlows not only achieve strong density estimation results on natural images such as MNIST, CIFAR-10, and ImageNet-32{\texttimes}32, but also obtain significantly better log-likelihoods on structured datasets such as galaxy images and MIMIC-III patient cohorts{—}all while being more efficient in terms of memory and computation than relevant baselines.

> normalizing flows 的新方法，stanford group的，但是感觉也没太大改进


14. **Robust Models Are More Interpretable Because Attributions Look Normal** -[PDF](https://proceedings.mlr.press/v162/wang22e.html)

> Recent work has found that adversarially-robust deep networks used for image classification are more interpretable: their feature attributions tend to be sharper, and are more concentrated on the objects associated with the image’s ground- truth class. We show that smooth decision boundaries play an important role in this enhanced interpretability, as the model’s input gradients around data points will more closely align with boundaries’ normal vectors when they are smooth. Thus, because robust models have smoother boundaries, the results of gradient- based attribution methods, like Integrated Gradients and DeepLift, will capture more accurate information about nearby decision boundaries. This understanding of robust interpretability leads to our second contribution: boundary attributions, which aggregate information about the normal vectors of local decision bound- aries to explain a classification outcome. We show that by leveraging the key fac- tors underpinning robust interpretability, boundary attributions produce sharper, more concentrated visual explanations{—}even on non-robust models.

> 有code，但是比较抽象


15. **Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling** -[PDF](https://proceedings.mlr.press/v162/nguyen22b.html)

> Neural Processes (NPs) are a popular class of approaches for meta-learning. Similar to Gaussian Processes (GPs), NPs define distributions over functions and can estimate uncertainty in their predictions. However, unlike GPs, NPs and their variants suffer from underfitting and often have intractable likelihoods, which limit their applications in sequential decision making. We propose Transformer Neural Processes (TNPs), a new member of the NP family that casts uncertainty-aware meta learning as a sequence modeling problem. We learn TNPs via an autoregressive likelihood-based objective and instantiate it with a novel transformer-based architecture that respects the inductive biases inherent to the problem structure, such as invariance to the observed data points and equivariance to the unobserved points. We further design knobs within the TNP architecture to tradeoff the increase in expressivity of the decoding distribution with extra computation. Empirically, we show that TNPs achieve state-of-the-art performance on various benchmark problems, outperforming all previous NP variants on meta regression, image completion, contextual multi-armed bandits, and Bayesian optimization.

> topic很有意思，有BO，还有GP， NN，之类的比较，涉及UQ + meta learning，但是是基于transformer 架构


16. **Double Sampling Randomized Smoothing** [PDF](https://proceedings.mlr.press/v162/li22aa.html)

> Neural networks (NNs) are known to be vulnerable against adversarial perturbations, and thus there is a line of work aiming to provide robustness certification for NNs, such as randomized smoothing, which samples smoothing noises from a certain distribution to certify the robustness for a smoothed classifier. However, as previous work shows, the certified robust radius in randomized smoothing suffers from scaling to large datasets ("curse of dimensionality"). To overcome this hurdle, we propose a Double Sampling Randomized Smoothing (DSRS) framework, which exploits the sampled probability from an additional smoothing distribution to tighten the robustness certification of the previous smoothed classifier. Theoretically, under mild assumptions, we prove that DSRS can certify Θ(𝑑‾‾√) robust radius under ℓ2 norm where 𝑑 is the input dimension, which implies that DSRS may be able to break the curse of dimensionality of randomized smoothing. We instantiate DSRS for a generalized family of Gaussian smoothing and propose an efficient and sound computing method based on customized dual optimization considering sampling error. Extensive experiments on MNIST, CIFAR-10, and ImageNet verify our theory and show that DSRS certifies larger robust radii than existing baselines consistently under different settings. Code is available at https://github.com/llylly/DSRS.

> 非常长，很理论，46页


17. **Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging** [PDF](https://proceedings.mlr.press/v162/angelopoulos22a.html) [Code](https://github.com/aangelopoulos/im2im-uq)

> Image-to-image regression is an important learning task, used frequently in biological imaging. Current algorithms, however, do not generally offer statistical guarantees that protect against a model’s mistakes and hallucinations. To address this, we develop uncertainty quantification techniques with rigorous statistical guarantees for image-to-image regression problems. In particular, we show how to derive uncertainty intervals around each pixel that are guaranteed to contain the true value with a user-specified confidence probability. Our methods work in conjunction with any base machine learning model, such as a neural network, and endow it with formal mathematical guarantees{—}regardless of the true unknown data distribution or choice of model. Furthermore, they are simple to implement and computationally inexpensive. We evaluate our procedure on three image-to-image regression tasks: quantitative phase microscopy, accelerated magnetic resonance imaging, and super-resolution transmission electron microscopy of a Drosophila melanogaster brain.

> 这个paper 非常UQ，但主要也是应用，MRI图片， 给了3个cases，

18. **Spectral Representation of Robustness Measures for Optimization Under Input Uncertainty** [PDF](https://proceedings.mlr.press/v162/qing22a.html)

> We study the inference of mean-variance robustness measures to quantify input uncertainty under the Gaussian Process (GP) framework. These measures are widely used in applications where the robustness of the solution is of interest, for example, in engineering design. While the variance is commonly used to characterize the robustness, Bayesian inference of the variance using GPs is known to be challenging. In this paper, we propose a Spectral Representation of Robustness Measures based on the GP’s spectral representation, i.e., an analytical approach to approximately infer both robustness measures for normal and uniform input uncertainty distributions. We present two approximations based on different Fourier features and compare their accuracy numerically. To demonstrate their utility and efficacy in robust Bayesian Optimization, we integrate the analytical robustness measures in three standard acquisition functions for various robust optimization formulations. We show their competitive performance on numerical benchmarks and real-life applications.





19. **Uncertainty Modeling in Generative Compressed Sensing** [PDF](https://proceedings.mlr.press/v162/zhang22ai/zhang22ai.pdf)

> Compressed sensing (CS) aims to recover a high-dimensional signal with structural priors from its low-dimensional linear measurements. Inspired by the huge success of deep neural networks in modeling the priors of natural signals, generative neural networks have been recently used to replace the hand-crafted structural priors in CS. However, the reconstruction capability of the generative model is fundamentally limited by the range of its generator, typically a small subset of the signal space of interest. To break this bottleneck and thus reconstruct those out-of-range signals, this paper presents a novel method called CS-BGM that can effectively expands the range of generator. Specifically, CS-BGM introduces uncertainties to the latent variable and parameters of the generator, while adopting the variational inference (VI) and maximum a posteriori (MAP) to infer them. Theoretical analysis demonstrates that expanding the range of generators is necessary for reducing the reconstruction error in generative CS. Extensive experiments show a consistent improvement of CS-BGM over the baselines.


20. **On the Practicality of Deterministic Epistemic Uncertainty** [PDF](https://proceedings.mlr.press/v162/postels22a/postels22a.pdf)

> A set of novel approaches for estimating epistemic uncertainty in deep neural networks with a single forward pass has recently emerged as a valid alternative to Bayesian Neural Networks. On the premise of informative representations, these deterministic uncertainty methods (DUMs) achieve strong performance on detecting out-of-distribution (OOD) data while adding negligible computational costs at inference time. However, it remains unclear whether DUMs are well calibrated and can seamlessly scale to real-world applications - both prerequisites for their practical deployment. To this end, we first provide a taxonomy of DUMs, and evaluate their calibration under continuous distributional shifts. Then, we extend them to semantic segmentation. We find that, while DUMs scale to realistic vision tasks and perform well on OOD detection, the practicality of current methods is undermined by poor calibration under distributional shifts.

> DUQ 那部分工作的延伸， 这个paper挺好的！


21. **Model-Value Inconsistency as a Signal for Epistemic Uncertainty** [PDF](https://icml.cc/virtual/2022/spotlight/17966)

> Using a model of the environment and a value function, an agent can construct many estimates of a state’s value, by unrolling the model for different lengths and bootstrapping with its value function. Our key insight is that one can treat this set of value estimates as a type of ensemble, which we call an implicit value ensemble (IVE). Consequently, the discrepancy between these estimates can be used as a proxy for the agent’s epistemic uncertainty; we term this signal model-value inconsistency or self-inconsistency for short. Unlike prior work which estimates uncertainty by training an ensemble of many models and/or value functions, this approach requires only the single model and value function which are already being learned in most model-based reinforcement learning algorithms. We provide empirical evidence in both tabular and function approximation settings from pixels that self-inconsistency is useful (i) as a signal for exploration, (ii) for acting safely under distribution shifts, and (iii) for robustifying value-based planning with a learned model.

> UQ for RL的, 这个topic挺难的，但是比较fit AAAI的topic 


22. **Blurs Behave Like Ensembles: Spatial Smoothings to Improve Accuracy, Uncertainty, and Robustness** [PDF](https://proceedings.mlr.press/v162/park22b.html)

> Neural network ensembles, such as Bayesian neural networks (BNNs), have shown success in the areas of uncertainty estimation and robustness. However, a crucial challenge prohibits their use in practice. BNNs require a large number of predictions to produce reliable results, leading to a significant increase in computational cost. To alleviate this issue, we propose spatial smoothing, a method that ensembles neighboring feature map points of convolutional neural networks. By simply adding a few blur layers to the models, we empirically show that spatial smoothing improves accuracy, uncertainty estimation, and robustness of BNNs across a whole range of ensemble sizes. In particular, BNNs incorporating spatial smoothing achieve high predictive performance merely with a handful of ensembles. Moreover, this method also can be applied to canonical deterministic neural networks to improve the performances. A number of evidences suggest that the improvements can be attributed to the stabilized feature maps and the smoothing of the loss landscape. In addition, we provide a fundamental explanation for prior works {—} namely, global average pooling, pre-activation, and ReLU6 {—} by addressing them as special cases of spatial smoothing. These not only enhance accuracy, but also improve uncertainty estimation and robustness by making the loss landscape smoother in the same manner as spatial smoothing. The code is available at https://github.com/xxxnell/spatial-smoothing.

> 不同的算例可以应用，还挺有意思的工作！


23. **Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation** [PDF](https://proceedings.mlr.press/v162/kuleshov22a.html)

> Accurate probabilistic predictions can be characterized by two properties{—}calibration and sharpness. However, standard maximum likelihood training yields models that are poorly calibrated and thus inaccurate{—}a 90% confidence interval typically does not contain the true outcome 90% of the time. This paper argues that calibration is important in practice and is easy to maintain by performing low-dimensional density estimation. We introduce a simple training procedure based on recalibration that yields calibrated models without sacrificing overall performance; unlike previous approaches, ours ensures the most general property of distribution calibration and applies to any model, including neural networks. We formally prove the correctness of our procedure assuming that we can estimate densities in low dimensions and we establish uniform convergence bounds. Our results yield empirical performance improvements on linear and deep Bayesian models and suggest that calibration should be increasingly leveraged across machine learning.

> 偏理论的，针对uncertainty 一个小的细节做的工作

24. **Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification** [PDF](https://proceedings.mlr.press/v162/schwinn22a.html)

> The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.

> 主要是CV的应用，能不能在NLP有应用？



25. **Learning Efficient and Robust Ordinary Differential Equations via Invertible Neural Networks** -[PDF](https://proceedings.mlr.press/v162/zhi22a.html)

> Advances in differentiable numerical integrators have enabled the use of gradient descent techniques to learn ordinary differential equations (ODEs), where a flexible function approximator (often a neural network) is used to estimate the system dynamics, given as a time derivative. However, these integrators can be unsatisfactorily slow and unstable when learning systems of ODEs from long sequences. We propose to learn an ODE of interest from data by viewing its dynamics as a vector field related to another base vector field via a diffeomorphism (i.e., a differentiable bijection), represented by an invertible neural network (INN). By learning both the INN and the dynamics of the base ODE, we provide an avenue to offload some of the complexity in modelling the dynamics directly on to the INN. Consequently, by restricting the base ODE to be amenable to integration, we can speed up and improve the robustness of integrating trajectories from the learned system. We demonstrate the efficacy of our method in training and evaluating benchmark ODE systems, as well as within continuous-depth neural networks models. We show that our approach attains speed-ups of up to two orders of magnitude when integrating learned ODEs.

> 有code，有example，是autonoous system 的有点意思！



26. **A FINE-GRAINED ANALYSIS ON DISTRIBUTION SHIFT** [PDF](https://openreview.net/pdf?id=Dl4LetuLdyK)

> 非常好的一个baseline工作，评价很高，想办法做点东西在这个基础上！ 这个领域也非常有前途我觉得 
Robustness to distribution shifts is critical for deploying machine learning models in the real world.  


27. **IN SEARCH OF LOST DOMAIN GENERALIZATION** [PDF](https://openreview.net/pdf?id=lQdXeXDoWtI)

> 另外一个baseline的工作， 关于domain generalization 这个现在挺火的，很多人做！


28. **BOOSTING RANDOMIZED SMOOTHING WITH VARIANCE REDUCED CLASSIFIERS** [pdf](https://openreview.net/pdf?id=mHu2vIds_-b) 

> 做variance reduction 用Gaussian random smoothing 的东西，解决classifer的， 比较ensemble的


29. **AUTOREGRESSIVE QUANTILE FLOWS FOR PREDICTIVE UNCERTAINTY ESTIMATION** [pdf](https://openreview.net/pdf?id=z1-I6rOKv1S)

> 这个paper看不太懂，不知道写的啥，做了好几个任务，包括time series forecasting啥的



30. **SAMPLE EFFICIENT DEEP REINFORCEMENT LEARNING VIA UNCERTAINTY ESTIMATION** [pdf](https://openreview.net/pdf?id=vrW3tvDfOJQ)

> sample DRL via UQ， 感觉很玄学，但是给了code，很难得！

31. **UNIFYING LIKELIHOOD-FREE INFERENCE WITH BLACK-BOX OPTIMIZATION AND BEYOND** [PDF](https://openreview.net/pdf?id=1HxTO6CTkz)

> 类似DOE的东西，但是讲了很多方法和对比的细节！不是很thoeretical，但是很多算法流程！


32. **ANOMALY TRANSFORMER: TIME SERIES ANOMALY DETECTION WITH ASSOCIATION DISCREPANCY** [pdf](https://openreview.net/pdf?id=LzQQ89U1qm_)

> 都用transformer做AD了


33. **VARIATIONAL METHODS FOR SIMULATION-BASED INFERENCE** [pdf](https://openreview.net/pdf?id=kZ0UYdhqkNY)

> 很经典的问题，但是用VI 去解决inference，不知道效果是否好很多？还是可以scalable？


34. **LEVERAGING UNLABELED DATA TO PREDICT OUT-OF-DISTRIBUTION PERFORMANCE** [pdf](https://openreview.net/pdf?id=o_HsiMPYh_x)

> ODD 用了大量的实验

35. **META LEARNING LOW RANK COVARIANCE FACTORS FOR ENERGY-BASED DETERMINISTIC UNCERTAINTY** [pdf](https://openreview.net/pdf?id=GQd7mXSPua)

> 偏理论的，对比deterministic UQ 的paper


36. **TRANSFORMERS CAN DO BAYESIAN INFERENCE** -[pdf](https://openreview.net/pdf?id=KSugKcbNf9) 

> 这个非常interesting！ 这个如果有发展，的确是一个路子，还能熟悉transfomer 去做UQ， 是不是很多基础的问题都可以解决呢？


### ✅ Causal inference 

1.  **Causal Conceptions of Fairness and their Consequences** -[Oral] [PDF](https://proceedings.mlr.press/v162/nilforoshan22a.html) 

> Recent work highlights the role of causality in designing equitable decision-making algorithms. It is not immediately clear, however, how existing causal conceptions of fairness relate to one another, or what the consequences are of using these definitions as design principles. Here, we first assemble and categorize popular causal definitions of algorithmic fairness into two broad families: (1) those that constrain the effects of decisions on counterfactual disparities; and (2) those that constrain the effects of legally protected characteristics, like race and gender, on decisions. We then show, analytically and empirically, that both families of definitions almost always—in a measure theoretic sense—result in strongly Pareto dominated decision policies, meaning there is an alternative, unconstrained policy favored by every stakeholder with preferences drawn from a large, natural class. For example, in the case of college admissions decisions, policies constrained to satisfy causal fairness definitions would be disfavored by every stakeholder with neutral or positive preferences for both academic preparedness and diversity. Indeed, under a prominent definition of causal fairness, we prove the resulting policies require admitting all students with the same probability, regardless of academic qualifications or group membership. Our results highlight formal limitations and potential adverse consequences of common mathematical notions of causal fairness.

> 比较理论，但是causal 很多，这个paper是outstanding paper， 很solid，但是40 pages， stanford， NYU and harvord 


2. **Matching Learned Causal Effects of Neural Networks with Domain Priors** [PDF](https://proceedings.mlr.press/v162/kancheti22a.html)

> A trained neural network can be interpreted as a structural causal model (SCM) that provides the effect of changing input variables on the model’s output. However, if training data contains both causal and correlational relationships, a model that optimizes prediction accuracy may not necessarily learn the true causal relationships between input and output variables. On the other hand, expert users often have prior knowledge of the causal relationship between certain input variables and output from domain knowledge. Therefore, we propose a regularization method that aligns the learned causal effects of a neural network with domain priors, including both direct and total causal effects. We show that this approach can generalize to different kinds of domain priors, including monotonicity of causal effect of an input variable on output or zero causal effect of a variable on output for purposes of fairness. Our experiments on twelve benchmark datasets show its utility in regularizing a neural network model to maintain desired causal effects, without compromising on accuracy. Importantly, we also show that a model thus trained is robust and gets improved accuracy on noisy inputs.

> 关于causal的工作，提前看看

3. **Validating Causal Inference Methods** -[PDF](https://proceedings.mlr.press/v162/parikh22a.html)

> The fundamental challenge of drawing causal inference is that counterfactual outcomes are not fully observed for any unit. Furthermore, in observational studies, treatment assignment is likely to be confounded. Many statistical methods have emerged for causal inference under unconfoundedness conditions given pre-treatment covariates, including propensity score-based methods, prognostic score-based methods, and doubly robust methods. Unfortunately for applied researchers, there is no ‘one-size-fits-all’ causal method that can perform optimally universally. In practice, causal methods are primarily evaluated quantitatively on handcrafted simulated data. Such data-generative procedures can be of limited value because they are typically stylized models of reality. They are simplified for tractability and lack the complexities of real-world data. For applied researchers, it is critical to understand how well a method performs for the data at hand. Our work introduces a deep generative model-based framework, Credence, to validate causal inference methods. The framework’s novelty stems from its ability to generate synthetic data anchored at the empirical distribution for the observed sample, and therefore virtually indistinguishable from the latter. The approach allows the user to specify ground truth for the form and magnitude of causal effects and confounding bias as functions of covariates. Thus simulated data sets are used to evaluate the potential performance of various causal estimation methods when applied to data similar to the observed sample. We demonstrate Credence’s ability to accurately assess the relative performance of causal estimation techniques in an extensive simulation study and two real-world data applications from Lalonde and Project STAR studies.


4. **OPTIMAL TRANSPORT FOR CAUSAL DISCOVERY** [pdf](https://openreview.net/pdf?id=qwBK94cP1y)

> 比较理论，提出一个functional causal models (FCMs)
 

### ✅ Generative models 

1. **Equivariant Diffusion for Molecule Generation in 3D**  - [Oral] [PDF](https://proceedings.mlr.press/v162/hoogeboom22a.html)

> This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and the efficiency at training time.

2. **Path-Gradient Estimators for Continuous Normalizing Flows**  -[Oral] [PDF](https://proceedings.mlr.press/v162/vaitl22a.html)

> Recent work has established a path-gradient estimator for simple variational Gaussian distributions and has argued that the path-gradient is particularly beneficial in the regime in which the variational distribution approaches the exact target distribution. In many applications, this regime can however not be reached by a simple Gaussian variational distribution. In this work, we overcome this crucial limitation by proposing a path-gradient estimator for the considerably more expressive variational family of continuous normalizing flows. We outline an efficient algorithm to calculate this estimator and establish its superior performance empirically.

> continuous flow + variational 需要check 有没有code， 不是很复杂

3. **Exploring and Exploiting Hubness Priors for High-Quality GAN Latent Sampling** [PDF](https://proceedings.mlr.press/v162/liang22b.html) [Code](https://github.com/Byronliang8/HubnessGANSampling)

> Despite the extensive studies on Generative Adversarial Networks (GANs), how to reliably sample high-quality images from their latent spaces remains an under-explored topic. In this paper, we propose a novel GAN latent sampling method by exploring and exploiting the hubness priors of GAN latent distributions. Our key insight is that the high dimensionality of the GAN latent space will inevitably lead to the emergence of hub latents that usually have much larger sampling densities than other latents in the latent space. As a result, these hub latents are better trained and thus contribute more to the synthesis of high-quality images. Unlike the a posterior "cherry-picking", our method is highly efficient as it is an a priori method that identifies high-quality latents before the synthesis of images. Furthermore, we show that the well-known but purely empirical truncation trick is a naive approximation to the central clustering effect of hub latents, which not only uncovers the rationale of the truncation trick, but also indicates the superiority and fundamentality of our method. Extensive experimental results demonstrate the effectiveness of the proposed method. Our code is available at: https://github.com/Byronliang8/HubnessGANSampling.




4. **Matching Normalizing Flows and Probability Paths on Manifolds** -[PDF](https://proceedings.mlr.press/v162/ben-hamu22a.html) 

> Continuous Normalizing Flows (CNFs) are a class of generative models that transform a prior distribution to a model distribution by solving an ordinary differential equation (ODE). We propose to train CNFs on manifolds by minimizing probability path divergence (PPD), a novel family of divergences between the probability density path generated by the CNF and a target probability density path. PPD is formulated using a logarithmic mass conservation formula which is a linear first order partial differential equation relating the log target probabilities and the CNF’s defining vector field. PPD has several key benefits over existing methods: it sidesteps the need to solve an ODE per iteration, readily applies to manifold data, scales to high dimensions, and is compatible with a large family of target paths interpolating pure noise and data in finite time. Theoretically, PPD is shown to bound classical probability divergences. Empirically, we show that CNFs learned by minimizing PPD achieve state-of-the-art results in likelihoods and sample quality on existing low-dimensional manifold benchmarks, and is the first example of a generative model to scale to moderately high dimensional manifolds.


> 还是CNFs，有code，比baseline那种


4. **Generative Flow Networks for Discrete Probabilistic Modeling** -[PDF](https://proceedings.mlr.press/v162/zhang22v.html)


> We present energy-based generative flow networks (EB-GFN), a novel probabilistic modeling algorithm for high-dimensional discrete data. Building upon the theory of generative flow networks (GFlowNets), we model the generation process by a stochastic data construction policy and thus amortize expensive MCMC exploration into a fixed number of actions sampled from a GFlowNet. We show how GFlowNets can approximately perform large-block Gibbs sampling to mix between modes. We propose a framework to jointly train a GFlowNet with an energy function, so that the GFlowNet learns to sample from the energy distribution, while the energy learns with an approximate MLE objective with negative samples from the GFlowNet. We demonstrate EB-GFN’s effectiveness on various probabilistic modeling tasks. Code is publicly available at https://github.com/zdhNarsil/EB_GFN.

> 有code， youshu bengio组的工作，挺有意思的


5. **Flow-based Recurrent Belief State Learning for POMDPs** -[PDF](https://proceedings.mlr.press/v162/chen22q.html)

> Partially Observable Markov Decision Process (POMDP) provides a principled and generic framework to model real world sequential decision making processes but yet remains unsolved, especially for high dimensional continuous space and unknown models. The main challenge lies in how to accurately obtain the belief state, which is the probability distribution over the unobservable environment states given historical information. Accurately calculating this belief state is a precondition for obtaining an optimal policy of POMDPs. Recent advances in deep learning techniques show great potential to learn good belief states. However, existing methods can only learn approximated distribution with limited flexibility. In this paper, we introduce the \textbf{F}l\textbf{O}w-based \textbf{R}ecurrent \textbf{BE}lief \textbf{S}tate model (FORBES), which incorporates normalizing flows into the variational inference to learn general continuous belief states for POMDPs. Furthermore, we show that the learned belief states can be plugged into downstream RL algorithms to improve performance. In experiments, we show that our methods successfully capture the complex belief states that enable multi-modal predictions as well as high quality reconstructions, and results on challenging visual-motor control tasks show that our method achieves superior performance and sample efficiency.

> NF 的一个简单应用


6. **Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations** -[PDF](https://proceedings.mlr.press/v162/jo22a.html)

> Generating graph-structured data requires learning the underlying distribution of graphs. Yet, this is a challenging problem, and the previous graph generative methods either fail to capture the permutation-invariance property of graphs or cannot sufficiently model the complex dependency between nodes and edges, which is crucial for generating real-world graphs such as molecules. To overcome such limitations, we propose a novel score-based generative model for graphs with a continuous-time framework. Specifically, we propose a new graph diffusion process that models the joint distribution of the nodes and edges through a system of stochastic differential equations (SDEs). Then, we derive novel score matching objectives tailored for the proposed diffusion process to estimate the gradient of the joint log-density with respect to each component, and introduce a new solver for the system of SDEs to efficiently sample from the reverse diffusion process. We validate our graph generation method on diverse datasets, on which it either achieves significantly superior or competitive performance to the baselines. Further analysis shows that our method is able to generate molecules that lie close to the training distribution yet do not violate the chemical valency rule, demonstrating the effectiveness of the system of SDEs in modeling the node-edge relationships.

> 和之前yang song那个paper 非常像，有code，比较理论，主要还是做graph的东西



7. **ANALYTIC-DPM: AN ANALYTIC ESTIMATE OF THE OPTIMAL REVERSE VARIANCE IN DIFFUSION PROBABILISTIC MODELS** [pdf](https://openreview.net/pdf?id=0xiJLKH-ufZ)

> ICLR 2022 best paper award - 提高inference的 score-based model or probabilistic diffusion models. 这个可以应用 有code
https://github.com/baofff/Analytic-DPM 挺牛逼的，可能是score-based model 的新SOTA



8. **ON PREDICTING GENERALIZATION USING GANS** [pdf](https://openreview.net/pdf?id=eW5R4Cek6y6)

> 用GAN做predicting generization 这个有点意思，能不能用score-based model 去做呢？ 比如probabilistic prediction generalization 
这个完全可以做的！ 比如用flow或者score-based去做！ 这个没有什么人搞感觉！有用啊！ 没有code现在，不过case都很简单！


9. **SCORE-BASED GENERATIVE MODELING WITH CRITICALLY-DAMPED LANGEVIN DIFFUSION** [pdf](https://openreview.net/pdf?id=CzceR82CYc)

> 做的非常solid，很牛逼，特别全面，但是现在感觉坑太深，不太好入了！做做应用还可以！

10. **RESPONSIBLE DISCLOSURE OF GENERATIVE MODELS USING SCALABLE FINGERPRINTING** [pdf](https://openreview.net/pdf?id=sOK-zS6WHB)

> 基于GAN的，但是比较fancy的！写的挺好的
 

11. **PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS** [pdf](https://openreview.net/pdf?id=TIdIXIpzhoI)

> fast sampling for diffusion models. Jonathan Ho from google brain 


12. **TACKLING THE GENERATIVE LEARNING TRILEMMA WITH DENOISING DIFFUSION GANS** [pdf](https://openreview.net/pdf?id=JprM0p-q0Co)

> 现在这个diffusion model 太卷了！ 所有的baseline都要比，还是去做应用比较好！



13. **MCMC SHOULD MIX: LEARNING ENERGY-BASED MODEL WITH NEURAL TRANSPORT LATENT SPACE MCMC** [pdf](https://openreview.net/pdf?id=4C93Qvn-tz) 

> MCMC + EBM 那个路子，UCLA 朱松纯组的

14. **A TALE OF TWO FLOWS: COOPERATIVE LEARNING OF LANGEVIN FLOW AND NORMALIZING FLOW TOWARD ENERGY-BASED MODEL**
- [pdf](https://openreview.net/pdf?id=31d5RLCUuXC)

> EBM + flow + langevin flow ，综合各种，但是太卷了，baseline要比很多


15. **GATSBI: GENERATIVE ADVERSARIAL TRAINING FOR SIMULATION-BASED INFERENCE** [pdf](https://openreview.net/pdf?id=kR1hC6j48Tp) 

>  simulation-based inference via GAN or other generative models 这个重点是simulation based


16. **AUTOREGRESSIVE DIFFUSION MODEL** - [pdf](https://openreview.net/pdf?id=Lm8T39vLDTE) 

> We introduce Autoregressive Diffusion Models (ARDMs), a model class encompassing and generalizing order-agnostic autoregressive models (Uria et al., 2014) and absorbing discrete diffusion (Austin et al., 2021),


17. **REVISITING FLOW GENERATIVE MODELS FOR GROUPWISE OUT-OF-DISTRIBUTION DETECTION** -[pdf](https://openreview.net/pdf?id=6y2KBh-0Fd9)

> 重新做flow for OOD， 用了two-points kolmogoro-smirnov 测试


### ✅ Multimodality (Vision, Speech, Lanuager, Graph)

1. **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** - [Oral] [PDF](https://proceedings.mlr.press/v162/baevski22a.html)

> While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

> MetaAI的工作，大活，估计很多人会follow


2. NATURAL LANGUAGE DESCRIPTIONS OF DEEP VISUAL FEATURES [PDF](https://openreview.net/pdf?id=NudBMY-tzDr)

> ICLR 2022 的 oral paper. 关于语言和vision 交叉的一个work


3. UNSUPERVISED VISION-LANGUAGE GRAMMAR INDUCTION WITH SHARED STRUCTURE MODELING [PDF](https://openreview.net/pdf?id=N0n_QyQ5lBF)

> 定义新的任务，paper写的挺好的。We introduce a new task, unsupervised vision-language (VL) grammar induction. Given an image-caption pair, the goal is to extract a shared hierarchical structure for both image and language simultaneously. 


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


4. **Self-Supervised Representation Learning via Latent Graph Prediction** -[PDF](https://proceedings.mlr.press/v162/xie22e.html)

> Self-supervised learning (SSL) of graph neural networks is emerging as a promising way of leveraging unlabeled data. Currently, most methods are based on contrastive learning adapted from the image domain, which requires view generation and a sufficient number of negative samples. In contrast, existing predictive models do not require negative sampling, but lack theoretical guidance on the design of pretext training tasks. In this work, we propose the LaGraph, a theoretically grounded predictive SSL framework based on latent graph prediction. Learning objectives of LaGraph are derived as self-supervised upper bounds to objectives for predicting unobserved latent graphs. In addition to its improved performance, LaGraph provides explanations for recent successes of predictive models that include invariance-based objectives. We provide theoretical analysis comparing LaGraph to related methods in different domains. Our experimental results demonstrate the superiority of LaGraph in performance and the robustness to decreasing of training sample size on both graph-level and node-level tasks.

> graph prediction 还是TAMU那个组




### ✅ Neuro-Symbolic 

1. **Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval** -[PDF](https://proceedings.mlr.press/v162/alon22a.html)

> Retrieval-based language models (R-LM) model the probability of natural language text by combining a standard language model (LM) with examples retrieved from an external datastore at test time. While effective, a major bottleneck of using these models in practice is the computationally costly datastore search, which can be performed as frequently as every time step. In this paper, we present RetoMaton - retrieval automaton - which approximates the datastore search, based on (1) saving pointers between consecutive datastore entries, and (2) clustering of entries into "states". This effectively results in a weighted finite automaton built on top of the datastore, instead of representing the datastore as a flat list. The creation of the automaton is unsupervised, and a RetoMaton can be constructed from any text collection: either the original training corpus or from another domain. Traversing this automaton at inference time, in parallel to the LM inference, reduces its perplexity by up to 1.85, or alternatively saves up to 83% of the nearest neighbor searches over 𝑘NN-LM (Khandelwal et al., 2020) without hurting perplexity. Our code and trained models are available at https://github.com/neulab/retomaton .


> 这个paper的github非常全，值得后续探索，如果要做这个方向的话！




2. **Neural-Symbolic Models for Logical Queries on Knowledge Graphs** - [PDF](https://proceedings.mlr.press/v162/zhu22c.html)

> Answering complex first-order logic (FOL) queries on knowledge graphs is a fundamental task for multi-hop reasoning. Traditional symbolic methods traverse a complete knowledge graph to extract the answers, which provides good interpretation for each step. Recent neural methods learn geometric embeddings for complex queries. These methods can generalize to incomplete knowledge graphs, but their reasoning process is hard to interpret. In this paper, we propose Graph Neural Network Query Executor (GNN-QE), a neural-symbolic model that enjoys the advantages of both worlds. GNN-QE decomposes a complex FOL query into relation projections and logical operations over fuzzy sets, which provides interpretability for intermediate variables. To reason about the missing links, GNN-QE adapts a graph neural network from knowledge graph completion to execute the relation projections, and models the logical operations with product fuzzy logic. Experiments on 3 datasets show that GNN-QE significantly improves over previous state-of-the-art models in answering FOL queries. Meanwhile, GNN-QE can predict the number of answers without explicit supervision, and provide visualizations for intermediate variables.

> Jian tang 那个组的工作，这方面的工作不多


3. **NODE-GAM:NEURAL GENERALIZED ADDITIVE MODEL FOR INTERPRETABLE DEEP LEARNING** -[pdf](https://openreview.net/pdf?id=g8NJR6fCCl8)

> GAM + neural 也算是增强可解释性！ paper写的挺好的，比较solid，找不出明显的问题


### ✅ Knowledge Graph 


### ✅ Decision making 

1. COMPARING DISTRIBUTIONS BY MEASURING DIFFERENCES THAT AFFECT DECISION MAKING [pdf](https://openreview.net/pdf?id=KB5onONJIAU)

> stefano Ermon 组的工作，偏理论，解决difference between two distributions 然后应用到climate change problem 


2. **BRIDGING THE GAP: PROVIDING POST-HOC SYMBOLIC EXPLANATIONS FOR SEQUENTIAL DECISION-MAKING PROBLEMS WITH INSCRUTABLE REPRESENTATIONS** [pdf](https://openreview.net/pdf?id=o-1v9hdSult)

> 比较复杂，看不出来做了什么， 估计没有code 

### ✅ NLP 

1. **LANGUAGE MODELING VIA STOCHASTIC PROCESSES** [pdf](https://openreview.net/pdf?id=pMQwKL1yctf)

> 用stochastic process 去model 语言， 挺solid的paper 


### ✅ Active learning + BOED + data augumentation 

1. **LOW-BUDGET ACTIVE LEARNING VIA WASSERSTEIN DISTANCE: AN INTEGER PROGRAMMING APPROACH**  [pdf](https://openreview.net/pdf?id=v8OlxjGn23S)

> active learning 的新工作，比较理论

2. **GENEDISCO: A BENCHMARK FOR EXPERIMENTAL DESIGN IN DRUG DISCOVERY** -[pdf](https://openreview.net/pdf?id=-w2oomO6qgc)

> 主要是active learning + RL 之类的，做的benchmark，这个方向现在很火，但是可做的topic就这么几个！ 有code 在github

3. **WHAT MAKES BETTER AUGMENTATION STRATEGIES? AUGMENT DIFFICULT BUT NOT TOO DIFFERENT** - [pdf](https://openreview.net/pdf?id=Ucx3DQbC9GH)

> NLP 的data augmentation 不知道是否好用

4. **DISCREPANCY-BASED ACTIVE LEARNING FOR DOMAIN ADAPTATION** - [pdf](https://openreview.net/pdf?id=p98WJxUC3Ca) 

> active learning for domain adaptation 现在active learning 各种新的东西出来，可能跟labeled data 有限有关




### ✅ optimal transport 

1. **LOSSY COMPRESSION WITH DISTRIBUTION SHIFT AS ENTROPY CONSTRAINED OPTIMAL TRANSPORT** [pdf](https://openreview.net/pdf?id=BRFWxcZfAdC)

> 这个optimal transport 感觉很多地方会用，ICLR的paper很多，这个和flow的区别？

2. **OPTIMAL TRANSPORT FOR LONG-TAILED RECOGNITION WITH LEARNABLE COST MATRIX** - [pdf](https://openreview.net/pdf?id=t98k9ePQQpn)

> 算是应用！这种应用还是可以发出paper的
