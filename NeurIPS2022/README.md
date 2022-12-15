
# AI/ML conference paper review 

这种语言+数据服务类型的，必将会被GPT3 取代？最大的问题是不信任！lack of confidence! 这个是一直要想办法解决的问题！

possible keywords

topic 没有人做的？
- generative model calibration, uncertainty for generative model



### ✅  Invited talk

- Conformal Prediction in 2022 [link](https://nips.cc/virtual/2022/invited-talk/55872), from [Emmanuel Candes](https://scholar.google.com/citations?user=nRQi4O8AAAAJ&hl=en)

### ✅  Outstanding paper

- Riemannian Score-Based Generative Modelling
> [link](https://openreview.net/forum?id=oDRQGo8I7P)

作图非常漂亮

### ✅  Oral papers 

- Kossen, Jannik, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. "Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation." arXiv preprint arXiv:2202.06881 (2022). 

> [code](https://github.com/jlko/active-surrogate-estimators), [link](https://arxiv.org/abs/2202.06881), [slides](https://nips.cc/virtual/2022/poster/54273)


关于active learning的工作，oxford他们新的工作




- Brunet, Marc-Etienne, Ashton Anderson, and Richard Zemel. "Implications of Model Indeterminacy for Explanations of Automated Decisions." In Advances in Neural Information Processing Systems. 

> [slides](https://nips.cc/virtual/2022/poster/53043), [paper](https://openreview.net/forum?id=LzbrVf-l0Xq)

这个paper来之financial 这个作者现在servicenow，关注一下， key words "automated decisions"



- Pearce, Tim, Jong-Hyeon Jeong, and Jun Zhu. "Censored Quantile Regression Neural Networks for Distribution-Free Survival Analysis." In Advances in Neural Information Processing Systems.

> [slides](https://nips.cc/virtual/2022/poster/55198), [paper](https://openreview.net/pdf?id=pGcTocvaZkJ), [code](https://github.com/jxzhangjhu/Censored_Quantile_Regression_NN)

> 也是uncertainty的，但是是quantile regression的思路，还是有人做这个！



- Vodrahalli, Kailas, Tobias Gerstenberg, and James Zou. "Uncalibrated Models Can Improve Human-AI Collaboration." arXiv preprint arXiv:2202.05983 (2022). 
> [slides](https://nips.cc/virtual/2022/poster/53892), [paper](https://arxiv.org/abs/2202.05983), [code](https://github.com/kailas-v/human-ai-interactions)


- Yu, Yaodong, Stephen Bates, Yi Ma, and Michael I. Jordan. "Robust Calibration with Multi-domain Temperature Scaling." arXiv preprint arXiv:2206.02757 (2022).
> [slides](https://nips.cc/virtual/2022/poster/55037), [paper](https://arxiv.org/pdf/2206.02757.pdf)

关于calibration最新的工作？ SOTA？很简单看上去，没太有理论


- Li, Shuang, Xavier Puig, Yilun Du, Clinton Wang, Ekin Akyurek, Antonio Torralba, Jacob Andreas, and Igor Mordatch. "Pre-trained language models for interactive decision-making." arXiv preprint arXiv:2202.01771 (2022).
> [slides](https://nips.cc/virtual/2022/poster/54484), [paper](https://arxiv.org/abs/2202.01771), [code](https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making)

挺有意思的工作，基于GPT3 的decision making 


- Nguyen, Thao, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. "Quality not quantity: On the interaction between dataset design and robustness of clip." arXiv preprint arXiv:2208.05516 (2022). 

> [slides](https://nips.cc/virtual/2022/poster/53131), [paper](https://arxiv.org/abs/2208.05516)
topic 挺有意思的，关于CLIP的robustness 


- Liu, Guan-Horng, Tianrong Chen, Oswin So, and Evangelos A. Theodorou. "Deep Generalized Schr\" odinger Bridge." arXiv preprint arXiv:2209.09893 (2022). 
> [slides](https://nips.cc/virtual/2022/poster/54873), [paper](https://openreview.net/pdf?id=fp33Nsh0O5), [code](https://github.com/jxzhangjhu/DeepGSB)

关于diffusion 和 schrodinger bridge的paper非常多！


- Schuster, Tal, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. "Confident adaptive language modeling." arXiv preprint arXiv:2207.07061 (2022). 
- [slides](https://nips.cc/virtual/2022/poster/53256), [paper](https://arxiv.org/abs/2207.07061), [code]
这个看上去挺promising的，没有code，估计短时间也够呛！



---

### Uncertainty 
> uncertainty, calibrate, confidence/confident, reliable, advice, automation, decision making  

- Ansari, Navid, Hans-Peter Seidel, Nima Vahidi Ferdowsi, and Vahid Babaei. "Autoinverse: Uncertainty Aware Inversion of Neural Networks." arXiv preprint arXiv:2208.13780 (2022). 
> 就是之前NA那个方法的延续，也可以发出来？ 太扯了吧! 看主题！！！！太老的主题注定发不出来！

- Einbinder, Bat-Sheva, Yaniv Romano, Matteo Sesia, and Yanfei Zhou. "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning." arXiv preprint arXiv:2205.05878 (2022). 
> [code](https://github.com/bat-sheva/conformal-learning) 

这个用conformal的方法，现在还是挺多人用这个的！

- Brophy, Jonathan, and Daniel Lowd. "Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees." arXiv preprint arXiv:2205.11412 (2022). 
> [slides](https://nips.cc/virtual/2022/poster/55260) 
这个是orgen那个work，主要做regression的


- Gruber, Sebastian Gregor, and Florian Buettner. "Better Uncertainty Calibration via Proper Scores for Classification and Beyond." In Advances in Neural Information Processing Systems. 
> [slides](https://nips.cc/virtual/2022/poster/53262)
偏理论，但是结合了calibration，uncertainty在classification and beyond


- Wagh, Neeraj, Jionghao Wei, Samarth Rawal, Brent M. Berry, and Yogatheesan Varatharajah. "Evaluating Latent Space Robustness and Uncertainty of EEG-ML Models under Realistic Distribution Shifts." In Advances in Neural Information Processing Systems. 
> [slides](https://nips.cc/virtual/2022/poster/52788)
纯应用，但是可以看看大概做了啥？


- UQGAN: A Unified Model for Uncertainty Quantification of Deep Classifiers trained via Conditional GANs. 
> [slides](https://nips.cc/virtual/2022/poster/54777), [code](https://github.com/jxzhangjhu/UQGAN)
这个可以仔细看看，如何讲这个story，比如做UQDiffusion 米有人做! topic有点陈旧，分不高，容易跪！


- Scalable Sensitivity and Uncertainty Analyses for Causal-Effect Estimates of Continuous-Valued Interventions 
> [slides](https://nips.cc/virtual/2022/poster/54488) 
做causal的， from Gal group，之前不少工作比较系统！Andrew Jesson [google](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=ElJ_fC4AAAAJ)


- Nonparametric Uncertainty Quantification for Single Deterministic Neural Network 
> [slides](https://nips.cc/virtual/2022/poster/53103)
做nonparametric的，也是single DUQ类似的工作，可以看看，重点是分数怎么样，topic 能不能过？

- On Uncertainty, Tempering, and Data Augmentation in Bayesian Classification 
> [slides](https://nips.cc/virtual/2022/poster/52923)， [Code](https://github.com/activatedgeek/understanding-bayesian-classification) 
NYU Andrew Wilson team 的work，还挺不错的，值得关注


- JAWS: Auditing Predictive Uncertainty Under Covariate Shift
> [slides](https://nips.cc/virtual/2022/poster/52881)
covariate shift 的相关工作，值得关注一下！ JHU那个team的

- Uncertainty Estimation for Multi-view Data: The Power of Seeing the Whole Picture  
> [slides](https://openreview.net/forum?id=9WJU4Lu2KTX)
偏应用，分数也有给的很低的，但是还是给accept了，选好AC挺重要的！看看有啥值得借鉴的不


- Semantic uncertainty intervals for disentangled latent spaces 
> [slides](https://nips.cc/virtual/2022/poster/53260)
这个和generative model的结合，现在很想做这一块的东西！


- Confidence-based Reliable Learning under Dual Noises
> [slides](https://nips.cc/virtual/2022/poster/54109)
清华Jun Zhu 他们的work，挺有意思的


- Online Bipartite Matching with Advice: Tight Robustness-Consistency Tradeoffs for the Two-Stage Model 
> [slides](https://nips.cc/virtual/2022/poster/52991) 
这种paper特别多，比较容易过？ 偏金融，fianncial？ 


- Pre-Trained Language Models for Interactive Decision-Making 
> [slides](https://nips.cc/virtual/2022/poster/54484)
这种decision making 的好多都和RL结合！




--- 
### Diffusion，或者和flow结合的？optimal transport 

- Improving Diffusion Models for Inverse Problems using Manifold Constraints 
> [slides](https://nips.cc/virtual/2022/poster/53565) 
太多人做inverse了，已经挤不进去了


- On Analyzing Generative and Denoising Capabilities of Diffusion-based Deep Generative Models 
> [slides](https://nips.cc/virtual/2022/poster/53410)
这个看上去挺有意思的， 可以关注一下，现在主流diffusion 任务估计是太多了

- Maximum Likelihood Training of Implicit Nonlinear Diffusion Model
> [slides](https://nips.cc/virtual/2022/poster/53575)
这个可以follow一下！

- On Translation and Reconstruction Guarantees of the Cycle-Consistent Generative Adversarial Networks
> [slides](https://nips.cc/virtual/2022/poster/54131)
纯理论 Cycle Consistency Loss  

- Mutual Information Divergence: A Unified Metric for Multimodal Generative Models 
> [slides](https://nips.cc/virtual/2022/poster/55019)
做metric，里面和CLIP结合的，做multimodal的


---
### Bayesian 




### controllable text generation

Poster
Tue 9:00	QUARK: Controllable Text Generation with Reinforced Unlearning
Ximing Lu · Sean Welleck · Jack Hessel · Liwei Jiang · Lianhui Qin · Peter West · Prithviraj Ammanabrolu · Yejin Choi	
Poster
Tue 14:00	Controllable Text Generation with Neurally-Decomposed Oracle
Tao Meng · Sidi Lu · Nanyun Peng · Kai-Wei Chang	
Poster
Wed 9:00	Diffusion-LM Improves Controllable Text Generation
Xiang Li · John Thickstun · Ishaan Gulrajani · Percy Liang · Tatsunori Hashimoto	
Poster
Thu 9:00	On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting
Tomasz Korbak · Hady Elsahar · Germán Kruszewski · Marc Dymetman 