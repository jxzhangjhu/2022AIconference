
# AI/ML conference paper review 

这种语言+数据服务类型的，必将会被GPT3 取代？最大的问题是不信任！lack of confidence! 这个是一直要想办法解决的问题！

possible keywords

- uncertainty, calibrate, confidence/confident, reliable
- diffusion
- advice 
- automation, decision making 


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


- Ansari, Navid, Hans-Peter Seidel, Nima Vahidi Ferdowsi, and Vahid Babaei. "Autoinverse: Uncertainty Aware Inversion of Neural Networks." arXiv preprint arXiv:2208.13780 (2022). 
> 就是之前NA那个方法的延续，也可以发出来？ 太扯了吧! 看主题！！！！太老的主题注定发不出来！

- Einbinder, Bat-Sheva, Yaniv Romano, Matteo Sesia, and Yanfei Zhou. "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning." arXiv preprint arXiv:2205.05878 (2022). 
> [code](https://github.com/bat-sheva/conformal-learning) 

这个用conformal的方法，现在还是挺多人用这个的！


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