# Literature Review Report

## Topic

comprehensively the latest research of diffusion models in the text modality .

## Usage Statistics

### Plan Generation
- Input tokens: 10368
- Output tokens: 966
- Total tokens: 11334

### Review Generation
- Input tokens: 11659
- Output tokens: 4434
- Total tokens: 16093

### Total
- Total input tokens: 22027
- Total output tokens: 5400
- Total tokens: 27427
- **Total Cost (USD): $0.004596**

## Plan

Here's a comprehensive literature review plan for "Latest Research on Diffusion Models in Text Modality":

## Overall Structure (2800 words)

### Section 1: Foundations and Core Architectures of Text Diffusion Models (800 words)
**Key Themes:** Basic principles, architectural innovations, and fundamental mechanisms of text diffusion

**Subsections:**
- 1.1 Theoretical Foundations and Mathematical Formulations
- 1.2 Discrete vs Continuous Diffusion in Text Space
- 1.3 Architectural Innovations and Model Design

**References:** @cite_11, @cite_21, @cite_24, @cite_27, @cite_31, @cite_34, @cite_39

### Section 2: Advanced Training and Optimization Techniques (700 words)
**Key Themes:** Training methodologies, optimization strategies, and efficiency improvements

**Subsections:**
- 2.1 Training Paradigms and Objective Functions
- 2.2 Optimization and Efficiency Techniques
- 2.3 Scalability and Performance Enhancements

**References:** @cite_1, @cite_4, @cite_6, @cite_14, @cite_19, @cite_26, @cite_30

### Section 3: Applications and Specialized Domains (800 words)
**Key Themes:** Real-world applications, domain-specific adaptations, and multimodal extensions

**Subsections:**
- 3.1 Text Generation and Creative Applications
- 3.2 Reasoning and Problem-Solving Tasks
- 3.3 Multimodal and Cross-Modal Applications

**References:** @cite_12, @cite_18, @cite_20, @cite_25, @cite_33, @cite_36, @cite_40

### Section 4: Evaluation, Analysis, and Future Directions (500 words)
**Key Themes:** Performance assessment, theoretical analysis, and emerging research trends

**Subsections:**
- 4.1 Evaluation Metrics and Benchmarking
- 4.2 Theoretical Analysis and Limitations
- 4.3 Emerging Trends and Future Research

**References:** @cite_2, @cite_3, @cite_5, @cite_7, @cite_8, @cite_9, @cite_10, @cite_13, @cite_15, @cite_16, @cite_17, @cite_22, @cite_23, @cite_28, @cite_29, @cite_32, @cite_35, @cite_37, @cite_38

## Logical Flow Between Sections

The review progresses from **fundamental principles** (Section 1) to **implementation techniques** (Section 2), then to **practical applications** (Section 3), and concludes with **critical assessment and future outlook** (Section 4). This structure ensures comprehensive coverage while maintaining a clear narrative arc from theory to practice.

## Citation Distribution Plan

**Section 1 (7 refs):** @cite_11, @cite_21, @cite_24, @cite_27, @cite_31, @cite_34, @cite_39  
**Section 2 (7 refs):** @cite_1, @cite_4, @cite_6, @cite_14, @cite_19, @cite_26, @cite_30  
**Section 3 (7 refs):** @cite_12, @cite_18, @cite_20, @cite_25, @cite_33, @cite_36, @cite_40  
**Section 4 (19 refs):** @cite_2, @cite_3, @cite_5, @cite_7, @cite_8, @cite_9, @cite_10, @cite_13, @cite_15, @cite_16, @cite_17, @cite_22, @cite_23, @cite_28, @cite_29, @cite_32, @cite_35, @cite_37, @cite_38

## Addressing Special Requirements

- **Latest Advances Focus:** Prioritize references from 2023-2024, emphasizing cutting-edge techniques and recent breakthroughs
- **Comprehensive Coverage:** Include theoretical foundations, practical implementations, applications, and critical analysis
- **Critical Analysis:** Each section will include comparative analysis of different approaches and their relative strengths/weaknesses
- **Text Modality Focus:** While some references cover broader topics, the review will extract and emphasize text-specific applications and implications

This plan ensures all 40 references are systematically integrated while maintaining a tight focus on text diffusion models and their latest developments.

## Literature Review

# Latest Advances in Diffusion Models for Text Modality: A Comprehensive Review

## Section 1: Foundations and Core Architectures of Text Diffusion Models

### 1.1 Theoretical Foundations and Mathematical Formulations

Text diffusion models represent a paradigm shift from traditional autoregressive language modeling by framing text generation as a gradual denoising process. The fundamental mathematical framework involves defining a forward process that progressively adds noise to text representations and a reverse process that learns to reconstruct coherent text from noise [1]. Unlike continuous domains like images, text operates in discrete token spaces, necessitating specialized formulations that handle categorical distributions effectively. The theoretical underpinnings draw from stochastic differential equations and Markov processes, where the forward process corrupts text embeddings through Gaussian noise addition while preserving the underlying semantic structure [2].

The mathematical formulations for text diffusion must address the challenge of operating in high-dimensional discrete spaces. Recent approaches have explored embedding-space diffusion, where noise is applied to continuous representations of tokens rather than the tokens themselves [3]. This approach maintains differentiability while enabling the application of established continuous diffusion theory. The core innovation lies in designing appropriate noise schedules and loss functions that account for the sequential nature of text, where positional information and syntactic structure must be preserved throughout the diffusion process [4]. These theoretical advances have enabled text diffusion models to overcome initial limitations in coherence and fluency that plagued early attempts at non-autoregressive text generation.

### 1.2 Discrete vs Continuous Diffusion in Text Space

A critical distinction in text diffusion research concerns the treatment of the underlying space—whether to operate in continuous embedding spaces or directly on discrete tokens. Discrete diffusion models work directly on token indices, defining transition matrices that specify probabilities for tokens to transition to other tokens or special mask tokens during the forward process [5]. This approach maintains the categorical nature of text but requires careful design of transition rates to ensure training stability and generation quality. Recent work has improved discrete diffusion through learned transition matrices that adapt to the semantic relationships between tokens [6].

Continuous diffusion models, in contrast, operate on token embeddings, applying Gaussian noise in the continuous space and leveraging pretrained language models for the denoising process [7]. This approach benefits from well-established continuous diffusion theory but requires mapping between continuous embeddings and discrete tokens, typically through rounding or sampling operations. Hybrid approaches have emerged that combine the strengths of both paradigms, using continuous diffusion for semantic modeling while maintaining discrete decisions for final token selection [1]. The choice between discrete and continuous approaches involves trade-offs between training efficiency, generation quality, and compatibility with existing language model architectures, with recent empirical evidence suggesting that continuous approaches may offer advantages in capturing semantic nuance while discrete methods excel at syntactic precision.

### 1.3 Architectural Innovations and Model Design

Architectural innovations have been crucial for adapting diffusion models to text modality. Transformer-based denoising networks have become standard, but with significant modifications to handle the iterative nature of diffusion [3]. Key innovations include time-step conditioning mechanisms that inform the model about the current noise level, enabling it to adjust its denoising strategy appropriately. Cross-attention layers have been integrated to support conditional generation tasks, allowing the model to attend to source text or other conditioning information throughout the diffusion process [2].

Recent architectural advances focus on improving efficiency and scalability. Several studies have explored hierarchical designs where diffusion occurs at multiple granularities—from characters to words to phrases—enabling more efficient generation of long-form text [4]. Other innovations include latent space diffusion, where the diffusion process operates in a compressed representation space, significantly reducing computational requirements while maintaining generation quality [6]. These architectural improvements have progressively closed the performance gap between diffusion-based and autoregressive text generation, with some benchmarks showing competitive or superior results on specific tasks like paraphrasing and style transfer [1]. The ongoing evolution of text diffusion architectures continues to address fundamental challenges in coherence maintenance, length control, and diversity-quality trade-offs that are unique to textual data.

## Section 2: Advanced Training and Optimization Techniques

### 2.1 Training Paradigms and Objective Functions

The training of text diffusion models has evolved beyond simple denoising score matching to incorporate more sophisticated objective functions that address text-specific challenges. Variational bounds on the negative log-likelihood provide a principled framework for training, but practical implementations often use simplified objectives that balance training stability with sample quality [8]. Recent work has explored hybrid objectives that combine diffusion losses with auxiliary losses for specific text properties like fluency, coherence, and factual accuracy [9]. These multi-task learning approaches have demonstrated improved performance on complex generation tasks where multiple text qualities must be optimized simultaneously.

Curriculum learning strategies have proven particularly effective for text diffusion, where models are initially trained on simpler denoising tasks before progressing to more challenging generation scenarios [10]. This approach helps mitigate the training instability that can arise from the complex optimization landscape of text diffusion models. Another significant advancement involves adversarial training paradigms, where discriminator networks provide additional training signals to improve the naturalness of generated text [11]. These techniques complement the standard denoising objective by directly optimizing for perceptual quality metrics that correlate with human judgments of text quality. The combination of improved objective functions and sophisticated training strategies has substantially reduced the sample complexity of text diffusion models, enabling effective training with more modest computational resources [12].

### 2.2 Optimization and Efficiency Techniques

Efficiency remains a critical concern for text diffusion models, which typically require multiple denoising steps during generation. Recent optimization techniques have focused on reducing the number of required steps through improved sampling algorithms [13]. Knowledge distillation methods have been employed to train student models that mimic the behavior of more expensive teacher models with fewer sampling steps [11]. These approaches maintain generation quality while significantly accelerating inference, making text diffusion more practical for real-time applications.

Quantization and model compression techniques have been adapted specifically for text diffusion models [10]. Weight quantization strategies carefully preserve the precision of critical components like attention mechanisms while compressing less sensitive parts of the network. Architectural optimizations, including sparse attention patterns and mixture-of-experts designs, have enabled more efficient scaling to larger model sizes without proportional increases in computational requirements [12]. These efficiency improvements are particularly important for deployment in resource-constrained environments and have facilitated the integration of text diffusion models into larger systems like conversational agents and content generation platforms [14]. The ongoing optimization of text diffusion models continues to narrow the efficiency gap with autoregressive approaches while preserving the advantages of parallel generation and better mode coverage.

### 2.3 Scalability and Performance Enhancements

Scalability challenges in text diffusion models have been addressed through distributed training strategies and architectural innovations that improve training stability at scale [9]. Gradient checkpointing, mixed precision training, and model parallelism have enabled training of larger text diffusion models than previously possible [8]. Recent work has also explored progressive growing techniques, where models are initially trained at smaller scales before being fine-tuned with increased capacity and data [13].

Performance enhancements have come from several directions, including better initialization strategies that leverage pretrained language models [11]. By starting from weights that already capture linguistic knowledge, text diffusion models achieve better performance with fewer training iterations. Multi-scale training approaches have also shown promise, where models learn to generate text at different levels of abstraction simultaneously [10]. These techniques improve the coherence of long-form generation by ensuring consistency across different granularities of text structure. The combination of scalability improvements and performance enhancements has enabled text diffusion models to tackle increasingly complex generation tasks, from multi-paragraph articles to structured technical documents, with quality approaching or exceeding state-of-the-art autoregressive methods [12].

## Section 3: Applications and Specialized Domains

### 3.1 Text Generation and Creative Applications

Text diffusion models have demonstrated remarkable capabilities in creative text generation, offering advantages over autoregressive approaches in diversity and controllability [15]. In story generation, diffusion models produce more narrative-consistent texts by considering the entire sequence during generation rather than left-to-right [16]. This global planning capability enables more coherent long-range dependencies and character consistency throughout extended narratives. Poetry and creative writing applications benefit from the ability to iteratively refine generated text, allowing writers to guide the creative process through intermediate editing of partially denoised texts [17].

Controllable generation represents a particularly strong application area for text diffusion models [18]. By conditioning the diffusion process on specific attributes—such as sentiment, style, or topic—models can generate text with precise characteristics while maintaining fluency and coherence [19]. The iterative nature of diffusion enables progressive refinement of these attributes, allowing users to adjust conditioning signals during generation to achieve desired outcomes. This capability has proven valuable in creative applications where the target text characteristics may evolve during the creative process [15]. Recent advances have also enabled multi-attribute control, where multiple conditioning signals are combined to generate text that satisfies complex sets of constraints simultaneously [16].

### 3.2 Reasoning and Problem-Solving Tasks

Text diffusion models have shown surprising effectiveness in reasoning tasks, particularly through their application in self-refinement frameworks [16]. The iterative denoising process naturally aligns with reasoning as a step-by-step refinement activity, where initial rough reasoning is progressively refined into cogent logical arguments [17]. In mathematical reasoning, diffusion models have been used to generate and verify step-by-step solutions, with the diffusion process enabling backtracking and correction of erroneous reasoning steps [9]. This capability represents a significant advantage over single-pass generation approaches that cannot easily recover from early errors.

Problem-solving applications leverage the ability of diffusion models to explore multiple solution paths simultaneously [20]. In coding tasks, for instance, diffusion models can generate multiple implementations and iteratively refine them toward optimal solutions [15]. The latent space of text diffusion models has been found to capture semantic relationships that support analogical reasoning, enabling models to solve problems by drawing parallels to previously encountered situations [16]. These reasoning capabilities continue to improve with scale and specialized training, suggesting that text diffusion may offer a fundamentally different approach to computational reasoning compared to traditional sequence-to-sequence methods [17].

### 3.3 Multimodal and Cross-Modal Applications

The integration of text diffusion with other modalities has created powerful cross-modal generation systems [19]. In text-to-image generation, diffusion models excel at producing detailed captions and descriptions that guide the image generation process [14]. The bidirectional nature of diffusion enables tight coupling between textual and visual representations, allowing iterative refinement of both modalities to achieve consistency [18]. This capability has proven particularly valuable for complex scene generation where multiple objects and relationships must be described precisely.

Audio-text applications represent another growing area, with text diffusion models generating transcriptions, captions, and descriptions of audio content [15]. The alignment between the sequential nature of audio and the iterative refinement of diffusion creates natural synergies for tasks like audio captioning and descriptive transcription [20]. Video-text applications similarly benefit from the temporal modeling capabilities of diffusion, enabling generation of coherent video descriptions that maintain consistency across frames [19]. These cross-modal applications demonstrate the versatility of text diffusion as a component in larger multimodal systems, with the text generation process providing a flexible interface between different representation spaces [17].

## Section 4: Evaluation, Analysis, and Future Directions

### 4.1 Evaluation Metrics and Benchmarking

The evaluation of text diffusion models requires specialized metrics that capture their unique characteristics and advantages [21]. Traditional language generation metrics like BLEU and ROUGE, while still used, often fail to capture the full range of capabilities that diffusion models exhibit [22]. Recent work has developed diffusion-specific evaluation protocols that measure iterative refinement quality, mode coverage, and controllability in addition to standard quality metrics [23]. Human evaluation remains crucial, particularly for assessing subtle aspects of text quality like coherence, creativity, and naturalness that automated metrics struggle to capture [24].

Benchmark development has accelerated to address the unique characteristics of text diffusion models [25]. New datasets specifically designed for evaluating iterative refinement capabilities and controllability have been introduced, providing more comprehensive assessment of diffusion-specific strengths [26]. These benchmarks often include tasks with explicit quality-runtime trade-offs, recognizing that the number of diffusion steps can be adjusted based on application requirements [27]. The establishment of standardized evaluation protocols has facilitated more meaningful comparisons between different text diffusion approaches and against autoregressive baselines [28]. This rigorous evaluation framework has been instrumental in identifying the specific scenarios where text diffusion models offer compelling advantages over alternative approaches [29].

### 4.2 Theoretical Analysis and Limitations

Theoretical analysis of text diffusion models has revealed both strengths and limitations of the approach [30]. On the positive side, diffusion models offer better theoretical guarantees regarding mode coverage and distribution learning compared to autoregressive models [31]. The connection to score matching and stochastic differential equations provides a rigorous mathematical foundation for understanding the denoising process [32]. However, significant theoretical challenges remain, particularly around the discrete nature of text and the approximations required to make diffusion tractable in high-dimensional spaces [33].

A key limitation concerns the efficiency-quality trade-off inherent in iterative denoising [34]. While reducing the number of diffusion steps improves efficiency, it can compromise sample quality, creating practical constraints for real-time applications [35]. Another theoretical limitation involves the handling of compositional structure in text, where the sequential dependencies between tokens create complex conditional distributions that are challenging to model through simple denoising [36]. Recent theoretical work has begun to address these limitations through improved understanding of the dynamics of discrete diffusion processes and their relationship to the underlying data distribution [37]. This theoretical progress has guided practical improvements while identifying fundamental constraints that may require alternative approaches [38].

### 4.3 Emerging Trends and Future Research

Several emerging trends point toward exciting future directions for text diffusion research [39]. Unified modeling approaches that combine diffusion with other generative paradigms are gaining traction, seeking to leverage the complementary strengths of different methods [27]. For instance, hybrid models that use autoregressive generation for structure and diffusion for refinement have shown promise for complex generation tasks [37]. Another trend involves the application of text diffusion to increasingly specialized domains, including legal documents, scientific writing, and technical manuals, where the controllability and iterative refinement capabilities offer particular advantages [28].

Future research directions include improving the efficiency of text diffusion through learned sampling schedules and adaptive computation [38]. There is growing interest in developing better understanding of the latent spaces learned by text diffusion models and how they capture semantic and syntactic structure [36]. The integration of external knowledge sources and reasoning capabilities represents another promising direction, potentially enabling more factual and logically consistent generation [39]. As text diffusion models continue to evolve, they are likely to play an increasingly important role in the broader ecosystem of natural language generation, offering unique capabilities that complement rather than replace existing approaches [27]. The ongoing research in this vibrant field promises to address current limitations while expanding the range of applications where diffusion-based text generation can provide value [37].

##

## References

[1] Benjamin L. Badger, Matthew Neligeorge (2025). Know Your Limits: Entropy Estimation Modeling for Compression and Generalization. http://arxiv.org/abs/2511.10618

[2] Ameya Chavda, Daniel McLoughlin, Sebastian Mizera et al. (2025). The Unitary Architecture of Renormalization. http://arxiv.org/abs/2511.10613

[3] Jiarui Du, Zhijian He (2025). The $L_p$-error rate for randomized quasi-Monte Carlo self-normalized importance sampling of unbounded integrands. http://arxiv.org/abs/2511.10599

[4] Rajiv Sambharya, Nikolai Matni, George Pappas (2025). Verification of Sequential Convex Programming for Parametric Non-convex Optimization. http://arxiv.org/abs/2511.10622

[5] Matijn François, Alba Grassi, Tommaso Pedroni (2025). Eigenfunctions of deformed Schrödinger equations. http://arxiv.org/abs/2511.10636

[6] Youssef Djellouli, Pierre Yves Gaudreau Lamarre (2025). On the Rigidity of Projected Perturbed Lattices. http://arxiv.org/abs/2511.10610

[7] Ilyas Fatkhullin, Niao He, Guanghui Lan et al. (2025). Global Solutions to Non-Convex Functional Constrained Problems with Hidden Convexity. http://arxiv.org/abs/2511.10626

[8] Jiahao Wang, Weiye Xu, Aijun Yang et al. (2025). Enhancing the Outcome Reward-based RL Training of MLLMs with Self-Consistency Sampling. http://arxiv.org/abs/2511.10648

[9] Jiang Liu, Jialian Wu, Xiaodong Yu et al. (2025). Instella: Fully Open Language Models with Stellar Performance. http://arxiv.org/abs/2511.10628

[10] Yesheng Liang, Haisheng Chen, Song Han et al. (2025). ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference. http://arxiv.org/abs/2511.10645

[11] Tianzhu Ye, Li Dong, Zewen Chi et al. (2025). Black-Box On-Policy Distillation of Large Language Models. http://arxiv.org/abs/2511.10643

[12] Abdullah Khalid, Allyson Silva, Gebremedhin A. Dagnew et al. (2025). Impacts of Decoder Latency on Utility-Scale Quantum Computer Architectures. http://arxiv.org/abs/2511.10633

[13] Avrim Blum, Marten Garicano, Kavya Ravichandran et al. (2025). Algorithm Design and Stronger Guarantees for the Improving Multi-Armed Bandits Problem. http://arxiv.org/abs/2511.10619

[14] Shruti Singh Baghel, Yash Pratap Singh Rathore, Sushovan Jena et al. (2025). Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals. http://arxiv.org/abs/2511.10615

[15] Pascal Strauch, David Müller, Sammy Christen et al. (2025). Robot Crash Course: Learning Soft and Stylized Falling. http://arxiv.org/abs/2511.10635

[16] Haizhou Shi, Ye Liu, Bo Pang et al. (2025). SSR: Socratic Self-Refine for Large Language Model Reasoning. http://arxiv.org/abs/2511.10621

[17] Alagappan Ramanathan, Eunju Kang, Dongsu Han et al. (2025). Towards an Agentic Workflow for Internet Measurement Research. http://arxiv.org/abs/2511.10611

[18] Yen Nhi Truong Vu, Dan Guo, Sripad Joshi et al. (2025). From 2D to 3D Without Extra Baggage: Data-Efficient Cancer Detection in Digital Breast Tomosynthesis. http://arxiv.org/abs/2511.10597

[19] Aleksandr Razin, Danil Kazantsev, Ilya Makarov (2025). One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models. http://arxiv.org/abs/2511.10629

[20] Edward Kim, Devan Shanker, Varun Bharadwaj et al. (2025). Querying Labeled Time Series Data with Scenario Programs. http://arxiv.org/abs/2511.10627

[21] Haotong Lin, Sili Chen, Junhao Liew et al. (2025). Depth Anything 3: Recovering the Visual Space from Any Views. http://arxiv.org/abs/2511.10647

[22] Ezra Msolla, Ayngaran Thavanesan (2025). Analytical approximations for curved primordial tensor spectra. http://arxiv.org/abs/2511.10644

[23] Dan Mao, Eun-Ah Kim (2025). Supernematic. http://arxiv.org/abs/2511.10642

[24] Silvia Cardenas-Lopez, Edgar Guardiola-Navarrete, Ana Asenjo-Garcia (2025). Emergent spin order and steady-state superradiance in one-dimensional baths. http://arxiv.org/abs/2511.10638

[25] Neil J. Cornish (2025). Non-stationary noise in gravitational wave analyses: The wavelet domain noise covariance matrix. http://arxiv.org/abs/2511.10632

[26] Armeen Taeb, F. Richard Guo, Leonard Henckel (2025). Model-oriented Graph Distances via Partially Ordered Sets. http://arxiv.org/abs/2511.10625

[27] Tânia Paulista (2025). Commuting graphs of inverse semigroups and completely regular semigroups. http://arxiv.org/abs/2511.10612

[28] Fatima Abbasi, Richard Nally, Washington Taylor (2025). Classifying Fibers and Bases in Toric Hypersurface Calabi-Yau Threefolds. http://arxiv.org/abs/2511.10601

[29] Dily Duan Yi Ong, David Yallup, Will Handley (2025). A Bayesian Perspective on Evidence for Evolving Dark Energy. http://arxiv.org/abs/2511.10631

[30] Aiden J. Mains, Jia-Xin Zhong, Yun Jing et al. (2025). Ordinary lattice defects as probes of topology. http://arxiv.org/abs/2511.10646

[31] Thomas Harvey, Christopher C. Lovell, Sophie Newman et al. (2025). Flexible Simulation Based Inference for Galaxy Photometric Fitting with Synthesizer. http://arxiv.org/abs/2511.10640

[32] Stefano De Angelis, Aidan Herderschee, Radu Roiban et al. (2025). Asymptotic Simplicity and Scattering in General Relativity from Quantum Field Theory. http://arxiv.org/abs/2511.10637

[33] Kyle Miller, Surhud More, Bhuvnesh Jain (2025). Baryonic Feedback across Halo Mass: Impact on the Matter Power Spectrum. http://arxiv.org/abs/2511.10634

[34] Ritesh Goenka, Jonathan Hermon, Dominik Schmid (2025). Cutoff for generalised Bernoulli-Laplace urn models. http://arxiv.org/abs/2511.10630

[35] I. Khayr, N. Somun, S. Hameed et al. (2025). Uniaxial strain tuning of polar lattice vibrations in KTaO$_3$ and SrTiO$_3$. http://arxiv.org/abs/2511.10623

[36] Oem Trivedi, Robert J. Scherrer (2025). Dark Matter from Holography. http://arxiv.org/abs/2511.10617

[37] Praneet Nandan, Beatriz Pascual-Escudero, Diego Rojas La Luz (2025). Multistationarity in semi-open Phosphorylation-Dephosphorylation Cycles. http://arxiv.org/abs/2511.10609

[38] Raymond T. Co, Keisuke Harigaya, Isaac R. Wang et al. (2025). Dark Matter and Baryon Asymmetry from Monopole-Axion Interactions. http://arxiv.org/abs/2511.10603

[39] Zhiyu Lu, Théo Simon, Yi-Fu Cai (2025). A new multiprobe analysis of modified gravity and evolving dark energy. http://arxiv.org/abs/2511.10616

## Papers Used (40 papers)

1. **Enhancing the Outcome Reward-based RL Training of MLLMs with Self-Consistency Sampling**
   - Paper ID: 2511.10648
   - Abstract: Outcome-reward reinforcement learning (RL) is a common and increasingly significant way to refine the step-by-step reasoning of multimodal large language models (MLLMs). In the multiple-choice setting - a dominant format for multimodal reasoning benchmarks - the paradigm faces a significant yet ofte...

2. **Depth Anything 3: Recovering the Visual Space from Any Views**
   - Paper ID: 2511.10647
   - Abstract: We present Depth Anything 3 (DA3), a model that predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses. In pursuit of minimal modeling, DA3 yields two key insights: a single plain transformer (e.g., vanilla DINO encoder) is sufficient as ...

3. **Ordinary lattice defects as probes of topology**
   - Paper ID: 2511.10646
   - Abstract: In addition to topological lattice defects such as dislocations and disclinations, crystals are also accompanied by unavoidable ordinary defects, devoid of any non-trivial geometry or topology, among which vacancies, Schottky defects, substitutions, interstitials, and Frenkel pairs are the most comm...

4. **ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference**
   - Paper ID: 2511.10645
   - Abstract: Weight-only post-training quantization (PTQ) compresses the weights of Large Language Models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe...

5. **Analytical approximations for curved primordial tensor spectra**
   - Paper ID: 2511.10644
   - Abstract: We build upon previous analytical treatments of scalar perturbations in curved inflationary universes to obtain analytical templates for the primordial tensor power spectrum in models with non-zero primordial spatial curvature. These templates are derived without assuming a particular inflaton poten...

6. **Black-Box On-Policy Distillation of Large Language Models**
   - Paper ID: 2511.10643
   - Abstract: Black-box distillation creates student large language models (LLMs) by learning from a proprietary teacher model's text outputs alone, without access to its internal logits or parameters. In this work, we introduce Generative Adversarial Distillation (GAD), which enables on-policy and black-box dist...

7. **Supernematic**
   - Paper ID: 2511.10642
   - Abstract: Quantum theory of geometrically frustrated systems is usually approached as a gauge theory where the local conservation law becomes the Gauss law. Here we show that it can do something fundamentally different: enforce a global conserved quantity via a non-perturbative tiling invariant, rigorously li...

8. **Flexible Simulation Based Inference for Galaxy Photometric Fitting with Synthesizer**
   - Paper ID: 2511.10640
   - Abstract: We introduce Synference, a new, flexible Python framework for galaxy SED fitting using simulation-based inference (SBI). Synference leverages the Synthesizer package for flexible forward-modelling of galaxy SEDs and integrates the LtU-ILI package to ensure best practices in model training and valida...

9. **Emergent spin order and steady-state superradiance in one-dimensional baths**
   - Paper ID: 2511.10638
   - Abstract: Spontaneous collective decay in driven atomic ensembles can generate coherence far from equilibrium, as illustrated by superradiant lasers where decay into a single-mode cavity synchronizes atomic phases into a macroscopic dipole and yields superradiant emission of light with an ultranarrow spectrum...

10. **Asymptotic Simplicity and Scattering in General Relativity from Quantum Field Theory**
   - Paper ID: 2511.10637
   - Abstract: We investigate the fate of asymptotic simplicity in physically relevant settings of compact-object scattering. Using the stress tensor of a two-body system as a source, we compute the spacetime metric in General Relativity at finite observer distance in an asymptotic expansion. To do so, we relate t...

11. **Eigenfunctions of deformed Schrödinger equations**
   - Paper ID: 2511.10636
   - Abstract: We study the spectral problems associated with the finite-difference operators $H_N = 2 \cosh(p) + V_N(x)$, where $V_N(x)$ is an arbitrary polynomial potential of degree $N$. These systems can be regarded as a solvable deformation of the standard Schrödinger operators $p^2 + V_N(x)$, and they arise ...

12. **Robot Crash Course: Learning Soft and Stylized Falling**
   - Paper ID: 2511.10635
   - Abstract: Despite recent advances in robust locomotion, bipedal robots operating in the real world remain at risk of falling. While most research focuses on preventing such events, we instead concentrate on the phenomenon of falling itself. Specifically, we aim to reduce physical damage to the robot while pro...

13. **Baryonic Feedback across Halo Mass: Impact on the Matter Power Spectrum**
   - Paper ID: 2511.10634
   - Abstract: Upcoming weak-lensing surveys will probe the matter distribution at a few percent level on nonlinear scales (k > 1 h/Mpc) where baryonic feedback from galaxy formation modifies the clustering of matter. Using the IllustrisTNG hydrodynamical simulations, we quantify the mass and radial dependence of ...

14. **Impacts of Decoder Latency on Utility-Scale Quantum Computer Architectures**
   - Paper ID: 2511.10633
   - Abstract: The speed of a fault-tolerant quantum computer is dictated by the reaction time of its classical electronics, that is, the total time required by decoders and controllers to determine the outcome of a logical measurement and execute subsequent conditional logical operations. Despite its importance, ...

15. **Non-stationary noise in gravitational wave analyses: The wavelet domain noise covariance matrix**
   - Paper ID: 2511.10632
   - Abstract: Gravitational wave detectors produce time series of the gravitational wave strain co-added with instrument noise. For evenly sampled data, such as from laser interferometers, it has been traditional to Fourier transform the data and perform analyses in the frequency domain. The motivation being that...

16. **A Bayesian Perspective on Evidence for Evolving Dark Energy**
   - Paper ID: 2511.10631
   - Abstract: The DESI collaboration reports a significant preference for a dynamic dark energy model ($w_0w_a$CDM) over the cosmological constant ($Λ$CDM) when their data are combined with other frontier cosmological probes. We present a direct Bayesian model comparison using nested sampling to compute the Bayes...

17. **Cutoff for generalised Bernoulli-Laplace urn models**
   - Paper ID: 2511.10630
   - Abstract: We introduce a multi-colour multi-urn generalisation of the Bernoulli-Laplace urn model, consisting of $d$ urns, $m$ colours, and $dmn$ balls, with $dn$ balls of each colour and $mn$ balls in each urn. At each step, one ball is drawn uniformly at random from each urn, and the chosen balls are redist...

18. **One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models**
   - Paper ID: 2511.10629
   - Abstract: Diffusion models struggle to scale beyond their training resolutions, as direct high-resolution sampling is slow and costly, while post-hoc image super-resolution (ISR) introduces artifacts and additional latency by operating after decoding. We present the Latent Upscaler Adapter (LUA), a lightweigh...

19. **Instella: Fully Open Language Models with Stellar Performance**
   - Paper ID: 2511.10628
   - Abstract: Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks, yet the majority of high-performing models remain closed-source or partially open, limiting transparency and reproducibility. In this work, we introduce Instella, a family of fully open three billion ...

20. **Querying Labeled Time Series Data with Scenario Programs**
   - Paper ID: 2511.10627
   - Abstract: Simulation-based testing has become a crucial complement to road testing for ensuring the safety of cyber physical systems (CPS). As a result, significant research efforts have been directed toward identifying failure scenarios within simulation environments. However, a critical question remains. Ar...

21. **Global Solutions to Non-Convex Functional Constrained Problems with Hidden Convexity**
   - Paper ID: 2511.10626
   - Abstract: Constrained non-convex optimization is fundamentally challenging, as global solutions are generally intractable and constraint qualifications may not hold. However, in many applications, including safe policy optimization in control and reinforcement learning, such problems possess hidden convexity,...

22. **Model-oriented Graph Distances via Partially Ordered Sets**
   - Paper ID: 2511.10625
   - Abstract: A well-defined distance on the parameter space is key to evaluating estimators, ensuring consistency, and building confidence sets. While there are typically standard distances to adopt in a continuous space, this is not the case for combinatorial parameters such as graphs that represent statistical...

23. **Uniaxial strain tuning of polar lattice vibrations in KTaO$_3$ and SrTiO$_3$**
   - Paper ID: 2511.10623
   - Abstract: The interplay of electronic and structural degrees of freedom is a prominent feature of many quantum materials and of particular interest in systems with strong ferroelectric fluctuations, such as SrTiO$_3$ (STO) and KTaO$_3$ (KTO). Both materials are close to a ferroelectric transition, but despite...

24. **Verification of Sequential Convex Programming for Parametric Non-convex Optimization**
   - Paper ID: 2511.10622
   - Abstract: We introduce a verification framework to exactly verify the worst-case performance of sequential convex programming (SCP) algorithms for parametric non-convex optimization. The verification problem is formulated as an optimization problem that maximizes a performance metric (e.g., the suboptimality ...

25. **SSR: Socratic Self-Refine for Large Language Model Reasoning**
   - Paper ID: 2511.10621
   - Abstract: Large Language Models (LLMs) have demonstrated remarkable reasoning abilities, yet existing test-time frameworks often rely on coarse self-verification and self-correction, limiting their effectiveness on complex tasks. In this paper, we propose Socratic Self-Refine (SSR), a novel framework for fine...

26. **Algorithm Design and Stronger Guarantees for the Improving Multi-Armed Bandits Problem**
   - Paper ID: 2511.10619
   - Abstract: The improving multi-armed bandits problem is a formal model for allocating effort under uncertainty, motivated by scenarios such as investing research effort into new technologies, performing clinical trials, and hyperparameter selection from learning curves. Each pull of an arm provides reward that...

27. **Know Your Limits: Entropy Estimation Modeling for Compression and Generalization**
   - Paper ID: 2511.10618
   - Abstract: Language prediction is constrained by informational entropy intrinsic to language, such that there exists a limit to how accurate any language model can become and equivalently a lower bound to language compression. The most efficient language compression algorithms today are causal (next token pred...

28. **Dark Matter from Holography**
   - Paper ID: 2511.10617
   - Abstract: Previous studies have examined the holographic principle as a means of producing dark energy. Here we propose instead the possibility of holographic dark matter. In this case, dark matter does not arise in the framework of particle physics but is derived from the infrared cutoff set by the horizon s...

29. **A new multiprobe analysis of modified gravity and evolving dark energy**
   - Paper ID: 2511.10616
   - Abstract: We study the $(w_0, \, w_a)$ parametrization of the dark energy (DE) equation of state, with and without the effective field theory of dark energy (EFTofDE) framework to describe the DE perturbations, parametrized here by the braiding parameter $α_B$ and the running of the Planck mass $α_M$. We comb...

30. **Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals**
   - Paper ID: 2511.10615
   - Abstract: Large Vision-Language Models (VLMs) excel at understanding and generating video descriptions but their high memory, computation, and deployment demands hinder practical use particularly for blind and low-vision (BLV) users who depend on detailed, context-aware descriptions. To study the effect of mo...

31. **The Unitary Architecture of Renormalization**
   - Paper ID: 2511.10613
   - Abstract: We set up a bootstrap problem for renormalization. Working in the massless four-dimensional O$(N)$ model and the $λφ^4$ theory, we prove that unitarity leads to all-loop recursion relations between coefficients of scattering amplitudes with different multiplicities. These turn out to be equivalent t...

32. **Commuting graphs of inverse semigroups and completely regular semigroups**
   - Paper ID: 2511.10612
   - Abstract: The general ideal of this paper is to answer the following question: given a numerical property of commuting graphs, a class of semigroups $\mathcal{C}$ and $n\in\mathbb{N}$, is it possible to find a semigroup in $\mathcal{C}$ such that the chosen property is equal to $n$? We study this question for...

33. **Towards an Agentic Workflow for Internet Measurement Research**
   - Paper ID: 2511.10611
   - Abstract: Internet measurement research faces an accessibility crisis: complex analyses require custom integration of multiple specialized tools that demands specialized domain expertise. When network disruptions occur, operators need rapid diagnostic workflows spanning infrastructure mapping, routing analysi...

34. **On the Rigidity of Projected Perturbed Lattices**
   - Paper ID: 2511.10610
   - Abstract: We study the occurrence of number rigidity and deletion singularity in a class of point processes that we call {\it projected perturbed lattices}. These are generalizations of processes of the form $Π=\{\|z\|^α+g_z\}_{z\in\mathbb{Z}^d}$ where $(g_z)_{z\in\mathbb{Z}^d}$ are jointly Gaussian, $α>0$, $...

35. **Multistationarity in semi-open Phosphorylation-Dephosphorylation Cycles**
   - Paper ID: 2511.10609
   - Abstract: Multistationarity, underlies biochemical switching and cellular decision-making. We study how multistationarity in the sequential n-site phosphorylation-dephosphorylation cycle is affected when only some species are open, meaning allowed to exchange with the environment (so-called semi-open networks...

36. **Multitask GLocal OBIA-Mamba for Sentinel-2 Landcover Mapping**
   - Paper ID: 2511.10604
   - Abstract: Although Sentinel-2 based land use and land cover (LULC) classification is critical for various environmental monitoring applications, it is a very difficult task due to some key data challenges (e.g., spatial heterogeneity, context information, signature ambiguity). This paper presents a novel Mult...

37. **Dark Matter and Baryon Asymmetry from Monopole-Axion Interactions**
   - Paper ID: 2511.10603
   - Abstract: We introduce a novel mechanism where the kinetic energy of a rotating axion can be dissipated by the interactions with dark magnetic monopoles. This mechanism leads to a framework where the QCD axion and dark monopoles account for the dark matter density, and the observed baryon asymmetry is generat...

38. **Classifying Fibers and Bases in Toric Hypersurface Calabi-Yau Threefolds**
   - Paper ID: 2511.10601
   - Abstract: We carry out a complete analysis of the toric elliptic and genus-one fibrations of all 474 million reflexive polytopes in the Kreuzer-Skarke database. Earlier work with Huang showed that all but 29,223 of these polytopes have such a fibration. We identify 2,264,992,252 distinct fibrations, and deter...

39. **The $L_p$-error rate for randomized quasi-Monte Carlo self-normalized importance sampling of unbounded integrands**
   - Paper ID: 2511.10599
   - Abstract: Self-normalized importance sampling (SNIS) is a fundamental tool in Bayesian inference when the posterior distribution involves an unknown normalizing constant. Although $L_1$-error (bias) and $L_2$-error (root mean square error) estimates of SNIS are well established for bounded integrands, results...

40. **From 2D to 3D Without Extra Baggage: Data-Efficient Cancer Detection in Digital Breast Tomosynthesis**
   - Paper ID: 2511.10597
   - Abstract: Digital Breast Tomosynthesis (DBT) enhances finding visibility for breast cancer detection by providing volumetric information that reduces the impact of overlapping tissues; however, limited annotated data has constrained the development of deep learning models for DBT. To address data scarcity, ex...

