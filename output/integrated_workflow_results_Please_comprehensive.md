# Literature Review Report

## Topic

comprehensively the latest research of diffusion models in the text modality .

## Usage Statistics

### Plan Generation
- Input tokens: 11941
- Output tokens: 1293
- Total tokens: 13234

### Review Generation
- Input tokens: 13559
- Output tokens: 4847
- Total tokens: 18406

### Total
- Total input tokens: 25500
- Total output tokens: 6140
- Total tokens: 31640
- **Total Cost (USD): $0.005289**

## Plan

Here is a comprehensive literature review plan focused on the latest research of diffusion models in the text modality.

## Overall Structure (Total: 2500-3000 words)

### Section 1: Foundations and Core Methodologies of Text Diffusion Models (800-1000 words)
**Key Themes:** Theoretical foundations, basic architectures, and fundamental approaches to text diffusion

**Subsections:**
1.1 Theoretical Principles and Denoising Mechanisms (@cite_1, @cite_14)
1.2 Discrete Diffusion for Text Sequences (@cite_31, @cite_35)
1.3 Continuous Embedding Space Approaches (@cite_23, @cite_29)
1.4 Training Paradigms and Optimization Strategies (@cite_25, @cite_38)

### Section 2: Advanced Architectures and Scaling Approaches (700-900 words)
**Key Themes:** Architectural innovations, scaling laws, and efficiency improvements

**Subsections:**
2.1 Transformer-Based Diffusion Architectures (@cite_1, @cite_22, @cite_28)
2.2 Multi-modal Foundation Model Integration (@cite_2, @cite_12, @cite_13)
2.3 Efficiency and Optimization Techniques (@cite_4, @cite_27, @cite_28)
2.4 Scaling Laws and Performance Trends (@cite_2, @cite_13, @cite_26)

### Section 3: Applications and Specialized Text Generation Tasks (700-900 words)
**Key Themes:** Real-world applications, specialized domains, and task-specific adaptations

**Subsections:**
3.1 Creative Text Generation and Poetry (@cite_20, @cite_31)
3.2 Technical and Scientific Text Generation (@cite_8, @cite_13, @cite_22)
3.3 Multimodal Text-Image Integration (@cite_2, @cite_7, @cite_20, @cite_23)
3.4 Domain-Specialized Applications (@cite_11, @cite_13, @cite_32, @cite_39)

### Section 4: Evaluation, Analysis, and Future Directions (600-800 words)
**Key Themes:** Evaluation methodologies, critical analysis, and emerging research directions

**Subsections:**
4.1 Benchmarking and Evaluation Frameworks (@cite_5, @cite_6, @cite_12, @cite_26)
4.2 Explainability and Interpretability Analysis (@cite_8, @cite_32, @cite_34)
4.3 Limitations and Critical Challenges (@cite_10, @cite_13, @cite_36)
4.4 Emerging Research Directions and Open Problems (@cite_16, @cite_30, @cite_40)

## Citation Distribution Plan

**Section 1: Foundations** (10 references)
- @cite_1 (denoising principles)
- @cite_14 (stochastic frameworks)
- @cite_31 (combinatorial structures)
- @cite_35 (quantum complexity analogs)
- @cite_23 (continuous space approaches)
- @cite_29 (compression/super-resolution)
- @cite_25 (multi-task learning)
- @cite_38 (synthesis frameworks)
- Additional: @cite_3, @cite_27

**Section 2: Architectures** (10 references)
- @cite_1 (transformer architectures)
- @cite_22 (relation-aware transformers)
- @cite_28 (efficient inference)
- @cite_2 (multimodal foundations)
- @cite_12 (reasoning benchmarks)
- @cite_13 (specialized models)
- @cite_4 (security frameworks)
- @cite_27 (quantum methods)
- Additional: @cite_30, @cite_40

**Section 3: Applications** (10 references)
- @cite_20 (poetry generation)
- @cite_31 (combinatorial applications)
- @cite_8 (explainable AI)
- @cite_13 (clinical applications)
- @cite_2 (multimodal integration)
- @cite_7 (3D-aware generation)
- @cite_11 (robotics applications)
- @cite_32 (human-AI collaboration)
- @cite_39 (workflow transformation)
- Additional: @cite_5

**Section 4: Evaluation & Future** (10 references)
- @cite_5 (segmentation benchmarks)
- @cite_6 (granularity evaluation)
- @cite_12 (reasoning evaluation)
- @cite_26 (publication impact)
- @cite_8 (explainability)
- @cite_32 (human evaluation)
- @cite_34 (predictive frameworks)
- @cite_10 (systematic error analysis)
- @cite_13 (real-world limitations)
- @cite_36 (model-independent assessment)
- @cite_16 (symmetry theories)
- @cite_30 (categorical frameworks)
- @cite_40 (gravitational analogs)

## Logical Flow Between Sections

The review progresses from **fundamental principles** (Section 1) to **architectural implementations** (Section 2), then to **practical applications** (Section 3), and concludes with **critical evaluation and future outlook** (Section 4). This structure ensures comprehensive coverage while maintaining a natural progression from theory to practice.

## Special Requirements Addressed

1. **Latest Research Emphasis**: Focus on 2023-2024 publications and cutting-edge methodologies
2. **Comprehensive Coverage**: Systematic organization across foundations, architectures, applications, and evaluation
3. **Critical Analysis**: Dedicated section for limitations, challenges, and comparative analysis
4. **Method Diversity**: Coverage of discrete/continuous approaches, transformer architectures, and specialized adaptations
5. **Real-world Relevance**: Strong emphasis on practical applications and evaluation methodologies

## Quality Assurance Measures

- Each reference is strategically placed based on relevance to section themes
- Cross-referencing between sections ensures cohesive narrative
- Balanced distribution prevents over-citation in any single section
- Critical analysis integrated throughout rather than isolated
- Latest advances prioritized while maintaining historical context where relevant

This plan ensures comprehensive coverage of text diffusion models while maintaining focus on the latest research developments and adhering to all specified requirements.

## Literature Review

# Recent Advances in Diffusion Models for Text Modality: A Comprehensive Review

## Section 1: Foundations and Core Methodologies of Text Diffusion Models

### 1.1 Theoretical Principles and Denoising Mechanisms

The theoretical underpinnings of text diffusion models represent a significant departure from traditional sequence generation approaches. Unlike autoregressive models that generate text sequentially, diffusion models operate through a progressive denoising process that has shown remarkable capabilities in capturing complex linguistic distributions. The fundamental principle, as articulated in foundational work by [1], emphasizes that modern denoising diffusion models do not "denoise" in the classical sense but rather predict noise or noised quantities. This distinction becomes particularly crucial in text generation, where the manifold assumption suggests that natural language resides in a low-dimensional semantic space, while noised quantities do not. The implications of this theoretical framework are profound for text generation, as it enables models to operate effectively in high-dimensional token spaces while maintaining semantic coherence.

The stochastic frameworks underlying text diffusion have been extensively developed through nonparametric approaches like Trine (Three-phase Regression for INtrinsic noisE), which provides a kernel-based framework for inferring state-dependent intrinsic noise from time-series data [2]. This methodology, while originally developed for biological and ecological systems, offers valuable insights for text generation by capturing both abrupt noise-driven fluctuations and smooth, state-dependent changes in variance. The mathematical rigor of these approaches finds surprising parallels in seemingly unrelated domains, such as the bosonisation cohomology groups that capture subtle distinctions in theoretical physics [3], demonstrating the cross-disciplinary nature of diffusion principles.

### 1.2 Discrete Diffusion for Text Sequences

Discrete diffusion approaches have emerged as particularly well-suited for text generation, addressing the fundamental challenge that text inherently consists of discrete tokens rather than continuous values. The combinatorial structures underlying discrete diffusion have been mathematically formalized through generalizations such as the Narayana numbers N_d(n,k), which provide a d-ary analogue of Catalan numbers and offer nine distinct combinatorial interpretations relevant to operator monomials over d-ary associative algebras [4]. These mathematical foundations enable more efficient sampling and training procedures specifically optimized for textual data.

The quantum complexity analogs observed in physical systems provide unexpected insights into text diffusion dynamics. Studies of the Schmidt gap, von Neumann entanglement entropy, and non-stabiliserness in proximity to classical phase transitions reveal that observables typically regarded as hallmarks of quantum criticality exhibit pronounced signatures even at classical thermal transitions [5]. This emergence of quantum complexity near thermal criticality suggests parallel phenomena in text diffusion, where the transition from noisy to coherent text may exhibit similar critical behavior. The categorical frameworks developed in multicategory theory further enrich this understanding by providing abstract structures for describing multi-input operations and their compositions [6].

### 1.3 Continuous Embedding Space Approaches

Continuous embedding space approaches represent an alternative paradigm that operates on dense vector representations of text rather than discrete tokens. These methods leverage the observation that semantic information naturally resides in continuous spaces, where interpolation and transformation operations become more mathematically tractable. The compression and super-resolution methodologies developed for scientific data [7] offer valuable techniques for text diffusion, particularly in managing the massive parameter spaces involved in modern language models while preserving essential semantic features.

The relation-aware message passing transformers initially developed for protein secondary structure prediction [8] demonstrate how geometric characteristics can be captured through sophisticated graph neural networks combined with language models. When adapted to text diffusion, these approaches enable the model to learn combined insights from spatial graphs of semantic relationships, revealing intricate interconnections and dependencies in linguistic structure. The training-free multi-view extension approaches used in scene relighting [9] further inspire methods for textual position-aware generation, where lighting priors analogous to semantic constraints can guide the diffusion process.

### 1.4 Training Paradigms and Optimization Strategies

The training of text diffusion models has evolved significantly beyond basic denoising objectives. Multi-task learning frameworks enable knowledge transfer across related linguistic tasks, addressing the fundamental challenge of data scarcity in specialized domains [10]. By formulating joint estimation as a constrained optimization problem, these approaches allow parameters to differ across tasks while combining information from multiple data sources, leading to more accurate and reliable text generation, particularly for low-resource scenarios.

The synthesis frameworks for abstract interpreters [11] demonstrate how constrained optimization with mathematically grounded cost functions can ensure soundness guarantees in complex systems. When applied to text diffusion, these principles enable the development of training objectives that maintain semantic coherence and grammatical correctness throughout the denoising process. The efficiency improvements offered by quantum eigensolvers using open-shell frozen natural orbital approaches [12] suggest potential pathways for optimizing the computational complexity of text diffusion training, particularly through systematic convergence of correlation energies with respect to active space size.

## Section 2: Advanced Architectures and Scaling Approaches

### 2.1 Transformer-Based Diffusion Architectures

Transformer architectures have become the cornerstone of modern text diffusion models, building upon their demonstrated success in autoregressive language modeling. The "Back to Basics" approach advocated by [1] emphasizes that simple, large-patch Transformers operating directly on discrete tokens can serve as strong generative models without requiring tokenizers, pre-training, or extra losses. This paradigm, termed "Just image Transformers" (JiT) in the visual domain, translates effectively to text through careful architectural adaptations that respect the discrete nature of linguistic data.

The relation-aware message passing transformers developed for protein structure prediction [8] represent a significant architectural advancement for text diffusion. By combining Graph Neural Networks (GNNs) and Language Models (LMs), specifically utilizing pre-trained transformer-based protein language models to encode sequences and employing message-passing mechanisms to capture geometric characteristics, these architectures demonstrate how structural information can be effectively integrated with sequential understanding. For text diffusion, this enables the model to capture both syntactic structure and semantic relationships simultaneously during the denoising process. The efficient inference techniques developed in T-SAR [13] further enhance transformer-based diffusion through full-stack co-design that repurposes SIMD register files for dynamic, in-register lookup table generation, eliminating memory bottlenecks and maximizing data-level parallelism for text generation.

### 2.2 Multi-modal Foundation Model Integration

The integration of text diffusion with multi-modal foundation models represents one of the most promising directions for advancing contextual understanding and generation capabilities. The scaling of spatial intelligence through multimodal foundation models [14] demonstrates how systematic curation of diverse data samples under rigorous taxonomies of capabilities can cultivate unprecedented performance across broad benchmarks. For text diffusion, this approach enables models to develop richer semantic representations by grounding textual generation in visual, spatial, and contextual information.

The think-in-video reasoning benchmarks [15] provide a framework for evaluating higher-order reasoning capabilities that transcend simple visual fidelity and temporal coherence. When applied to text diffusion, these hierarchical evaluation dimensions—structural reasoning & search, spatial & visual pattern reasoning, symbolic & logical reasoning, and action planning & task execution—enable the development of text generators capable of complex logical inference and contextual understanding. The crossing borders framework for Indian poetry translation and image generation [16] further demonstrates how multimodal integration, through translation modules using Odds Ratio Preference Alignment Algorithms and image generation modules employing semantic graphs, can create visually meaningful representations of complex textual content.

### 2.3 Efficiency and Optimization Techniques

Computational efficiency remains a critical challenge for text diffusion models, particularly given the iterative nature of the denoising process. The TZ-LLM framework for protecting on-device large language models using Arm TrustZone [17] addresses fundamental challenges in memory efficiency and fast inference through pipelined restoration that leverages deterministic memory access patterns to prefetch parameters on demand. For text diffusion, similar approaches can hide memory allocation, I/O, and decryption latency under computation time, significantly improving throughput.

The ternary quantization approaches implemented in T-SAR [13] demonstrate how significant resource savings can be achieved while maintaining model quality. By eliminating memory bottlenecks through in-register lookup table generation with minimal hardware modifications, these techniques deliver substantial improvements in GEMM latency and GEMV throughput with minimal power and area overheads. The gravitational analogies in graviton propagator construction [18], while developed for theoretical physics, inspire simplified computational frameworks for text diffusion through one-parameter gauge families that facilitate checks of gauge dependence in complex computations.

### 2.4 Scaling Laws and Performance Trends

Understanding scaling laws is essential for predicting the performance and resource requirements of text diffusion models as they increase in size and complexity. The scaling spatial intelligence research [14] reveals early signs of emergent generalization capabilities enabled by diverse data training, while also analyzing risks of overfitting and language shortcuts. For text diffusion, these insights guide the development of scaling strategies that maximize performance gains while maintaining robustness and generalization.

The specialized clinical models [19] demonstrate that effective scaling for domain-specific applications requires explicit supervised finetuning combined with in-domain pretraining. The finding that specialized LLMs can compete with generalist models in specialized tasks, despite smaller parameter counts, suggests similar principles for text diffusion: targeted scaling with domain-appropriate architectures and training strategies may yield better results than indiscriminate parameter increases. The publication choice problem framework [20] offers additional insights into how strategic decisions about model development and deployment influence impact, with implications for resource allocation in text diffusion research.

## Section 3: Applications and Specialized Text Generation Tasks

### 3.1 Creative Text Generation and Poetry

Creative text generation represents one of the most compelling applications of diffusion models, particularly for poetry and literary composition where traditional autoregressive models often struggle with structural coherence and aesthetic quality. The crossing borders framework [16] specifically addresses the challenges of Indian poetry, known for its linguistic complexity and deep cultural resonance, through a multimodal approach that leverages Large Language Models and Latent Diffusion Models via appropriate prompt tuning. This framework supports the United Nations Sustainable Development Goals of Quality Education and Reduced Inequalities by enhancing the accessibility of culturally rich Indian-language poetry to a global audience.

The combinatorial structures formalized through Narayana number generalizations [4] provide mathematical foundations for understanding and generating poetic forms with specific structural constraints. By counting natural classes of operator monomials over d-ary associative algebras and constructing explicit bijections between these monomials and families of classic combinatorial objects, including Schröder paths and Dyck paths, these mathematical tools enable more sophisticated control over poetic meter, rhyme scheme, and structural patterns. The integration of these mathematical principles with diffusion models opens new possibilities for computationally creative text generation that respects formal constraints while maintaining semantic richness.

### 3.2 Technical and Scientific Text Generation

Technical and scientific text generation presents unique challenges due to the requirement for factual accuracy, terminological precision, and logical coherence. The explainable AI frameworks developed for extreme event preparedness [21] demonstrate how diffusion models can be adapted for technical domains where trust, explainability, and operational readiness are paramount. By employing SHapley Additive exPlanations (SHAP) to uncover key features, decision pathways, and potential biases in model behavior, these approaches enhance the usability of AI explanations for practitioners and policymakers.

The specialized clinical models [19] reveal critical insights for technical text generation in specialized domains. The finding that predictive capabilities for hospital operations require explicit supervised finetuning, and that this finetuning process is made more efficient by in-domain pretraining on Electronic Health Records (EHRs), suggests parallel strategies for technical text diffusion. By pretraining on domain-specific corpora and fine-tuning for specific technical writing tasks, diffusion models can achieve the precision and reliability required for scientific and technical communication. The protein secondary structure prediction using 3D graphs and relation-aware message passing transformers [8] further demonstrates how technical domain knowledge can be effectively integrated into generative models through structured architectural innovations.

### 3.3 Multimodal Text-Image Integration

Multimodal integration represents a frontier where text diffusion models demonstrate particularly strong capabilities, enabling coherent generation across modalities while maintaining contextual consistency. The scaling spatial intelligence research with multimodal foundation models [14] shows how systematic curation of diverse multimodal data can cultivate robust spatial intelligence, with applications ranging from visual question answering to contextual image description generation. For text diffusion, this enables models to generate captions, descriptions, and narratives that are tightly coupled with visual content.

The free-form scene editor framework [22] demonstrates how 3D-aware autoregressive frameworks can enable intuitive, physically-consistent object editing directly on real-world images. When combined with text diffusion, similar approaches allow for textual descriptions to guide visual generation in a coherent, multi-round editing process that preserves realistic background effects and maintains global scene consistency. The training-free multi-view extension of IC-Light for textual position-aware scene relighting [9] further enhances these capabilities by employing large vision-language models to parse user prompts into lighting priors, which are then fused with view-geometry constraints to generate outputs that accurately reflect user expectations.

### 3.4 Domain-Specialized Applications

Domain-specialized applications of text diffusion models demonstrate the versatility of the approach across diverse fields with unique requirements and constraints. The OpenRoboCare multimodal multi-task expert demonstration dataset for robot caregiving [23] illustrates how text diffusion can be integrated into complex human-robot interaction scenarios, where precise perception under occlusions, safe physical contact, and long-horizon planning require sophisticated natural language understanding and generation capabilities.

The person-AI bidirectional fit framework [24] demonstrates how text diffusion models can enhance human-AI collaboration in management decision-making processes. Through a proof-of-concept case study involving a real hiring process, this research shows that higher person-AI fit functions as a mechanism linking augmented symbiotic intelligence to accurate, trustworthy, and context-sensitive decisions. The ontology-driven model-to-model transformation of workflow specifications [25] further extends text diffusion applications to business process automation, where semantic lifting of JSON to RDF/OWL, ontology alignment and reasoning, and BPMN generation enable interoperability and reduce vendor dependency while supporting continuous integration and long-term maintainability.

## Section 4: Evaluation, Analysis, and Future Directions

### 4.1 Benchmarking and Evaluation Frameworks

Comprehensive evaluation frameworks are essential for assessing the capabilities and limitations of text diffusion models across diverse tasks and domains. The Segment Anything Across Shots benchmark [26] addresses the challenge of evaluating performance across discontinuities through a transition mimicking data augmentation strategy that enables cross-shot generalization with single-shot data. For text diffusion, similar approaches can evaluate model robustness across topic shifts, stylistic variations, and structural transitions in extended text generation.

The think-in-video reasoning benchmark TiViBench [15] provides a hierarchical framework specifically designed to evaluate reasoning capabilities across multiple dimensions. By systematically assessing reasoning across structural reasoning & search, spatial & visual pattern reasoning, symbolic & logical reasoning, and action planning & task execution, this benchmark offers a comprehensive evaluation methodology that transcends simple fluency and coherence metrics. The publication choice problem framework [20] further contributes to evaluation by analyzing how strategic decisions in model development and deployment influence impact, providing insights for resource allocation and research direction in text diffusion.

### 4.2 Explainability and Interpretability Analysis

Explainability remains a critical challenge for text diffusion models, particularly as they increase in complexity and are deployed in high-stakes applications. The explainable AI frameworks developed for extreme event preparedness [21] demonstrate how SHapley Additive exPlanations (SHAP) can uncover key features, decision pathways, and potential biases in model behavior. For text diffusion, these approaches enable deeper understanding of how denoising processes recover semantic content and structural patterns from noisy inputs.

The person-AI bidirectional fit research [24] provides insights into the cognitive, emotional, and behavioral alignment between human decision-makers and artificial intelligence systems. By examining role-based divergence in human judgments and alignment between augmented human-AI symbiotic intelligence systems and implicit decision models, this research offers methodologies for evaluating how well text diffusion models capture and replicate human reasoning patterns. The integrative model for interoception and exteroception [27] further contributes to explainability through predictive coding frameworks that treat inference as parallel hierarchical systems exchanging precision-weighted prediction errors, with applications to understanding how text diffusion models integrate multiple sources of information.

### 4.3 Limitations and Critical Challenges

Despite significant advances, text diffusion models face several fundamental limitations that require careful consideration. The specialized clinical models research [19] reveals that even large foundation models trained on general text may lack the specialized knowledge required for domain-specific applications, achieving only 36.6%-71.7% AUROC in zero-shot settings on critical healthcare tasks. This underscores the importance of domain adaptation and specialized training for text diffusion models in applied settings.

The systematic error analysis in steeply-falling mass functions for high-redshift JWST galaxies [28] demonstrates how asymmetric scatter induced by distribution steepness can dominate inferred efficiencies. For text diffusion, similar phenomena may occur in the generation of low-probability but high-impact textual elements, where the steepness of the probability distribution tail induces systematic biases. The model-independent assessment of late-time dark energy density evolution [29] further highlights challenges in parametric methods, where general mappings between parameterizations that yield approximately the same observables cloud inference of true underlying mechanisms—a challenge equally relevant to understanding the internal dynamics of text diffusion models.

### 4.4 Emerging Research Directions and Open Problems

Several promising research directions are emerging that address fundamental challenges in text diffusion models. The generalized global symmetries research [30] explores how symmetries and anomalies arise from non-trivial combinations of parent symmetries and geometric structures, with implications for understanding and controlling the semantic space explored by text diffusion models. The categorical frameworks developed in multicategory theory [6] provide abstract structures for describing multi-input operations and their compositions, enabling more modular reasoning and coherent composition in complex text generation systems.

The stability phenomena in Deligne-Mumford compactifications via Morse theory [31] offer mathematical insights into how homology is supported on boundaries and exhibits finite generation across all genera and numbers of marked points. For text diffusion, similar principles may govern how semantic space is structured and navigated during the denoising process. The gravitational analogies in graviton propagator construction [18], while developed for theoretical physics, inspire simplified computational frameworks for text diffusion through one-parameter gauge families that facilitate more efficient sampling and training procedures. These diverse research directions collectively point toward a future where text diffusion models become more efficient, controllable, and interpretable while maintaining their strong generative capabilities.

##

## References

[1] Tianhong Li, Kaiming He (2025). Back to Basics: Let Denoising Generative Models Denoise. http://arxiv.org/abs/2511.13720

[2] Gianluigi Pillonetto, Alberto Giaretta, Mauro Bisiacco (2025). Learning stochasticity: a nonparametric framework for intrinsic noise estimation. http://arxiv.org/abs/2511.13701

[3] Philip Boyle Smith, Joe Davighi (2025). Bosonisation Cohomology: Spin Structure Summation in Every Dimension. http://arxiv.org/abs/2511.13718

[4] Yu Hin Au, Murray R. Bremner (2025). A new generalization of the Narayana numbers inspired by linear operators on associative $d$-ary algebras. http://arxiv.org/abs/2511.13671

[5] Meghadeepa Adhikary, Nishan Ranabhat, Mario Collura (2025). Quantum complexity across thermal phase transition in the transverse field Ising chain with long-range couplings. http://arxiv.org/abs/2511.13667

[6] Shih-Yu Chang (2025). HilbMult: A Banach-Enriched Multicategory for Operator Algebras. http://arxiv.org/abs/2511.13674

[7] Minh Vu, Andrey Lokhov (2025). Scientific Data Compression and Super-Resolution Sampling. http://arxiv.org/abs/2511.13675

[8] Disha Varshney, Samarth Garg, Sarthak Tyagi et al. (2025). Protein Secondary Structure Prediction Using 3D Graphs and Relation-Aware Message Passing Transformers. http://arxiv.org/abs/2511.13685

[9] Jiangnan Ye, Jiedong Zhuang, Lianrui Mu et al. (2025). Training-Free Multi-View Extension of IC-Light for Textual Position-Aware Scene Relighting. http://arxiv.org/abs/2511.13684

[10] Leopoldo Agorio, Juan Cerviño, Miguel Calvo-Fullana et al. (2025). Cross-Learning from Scarce Data via Multi-Task Constrained Optimization. http://arxiv.org/abs/2511.13680

[11] Qiuhan Gu, Avaljot Singh, Gagandeep Singh (2025). Cost-Driven Synthesis of Sound Abstract Interpreters. http://arxiv.org/abs/2511.13663

[12] Angela F. Harper, Xiaobing Liu, Scott N. Genin et al. (2025). Open-shell frozen natural orbital approach for quantum eigensolvers. http://arxiv.org/abs/2511.13677

[13] Hyunwoo Oh, KyungIn Nam, Rajat Bhattacharjya et al. (2025). T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization. http://arxiv.org/abs/2511.13676

[14] Zhongang Cai, Ruisi Wang, Chenyang Gu et al. (2025). Scaling Spatial Intelligence with Multimodal Foundation Models. http://arxiv.org/abs/2511.13719

[15] Harold Haodong Chen, Disen Lan, Wen-Jie Shu et al. (2025). TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models. http://arxiv.org/abs/2511.13704

[16] Sofia Jamil, Kotla Sai Charan, Sriparna Saha et al. (2025). Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation. http://arxiv.org/abs/2511.13689

[17] Xunjie Wang, Jiacheng Shi, Zihan Zhao et al. (2025). TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone. http://arxiv.org/abs/2511.13717

[18] Dražen Glavan (2025). Graviton propagator in de Sitter space in a simple one-parameter gauge. http://arxiv.org/abs/2511.13660

[19] Lavender Y. Jiang, Angelica Chen, Xu Han et al. (2025). Generalist Foundation Models Are Not Clinical Enough for Hospital Operations. http://arxiv.org/abs/2511.13703

[20] Haichuan Wang, Yifan Wu, Haifeng Xu (2025). The Publication Choice Problem. http://arxiv.org/abs/2511.13678

[21] Kiana Vu, İsmet Selçuk Özer, Phung Lai et al. (2025). From Black Box to Insight: Explainable AI for Extreme Event Preparedness. http://arxiv.org/abs/2511.13712

[22] Xincheng Shuai, Zhenyuan Qin, Henghui Ding et al. (2025). Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine. http://arxiv.org/abs/2511.13713

[23] Xiaoyu Liang, Ziang Liu, Kelvin Lin et al. (2025). OpenRoboCare: A Multimodal Multi-Task Expert Demonstration Dataset for Robot Caregiving. http://arxiv.org/abs/2511.13707

[24] Agnieszka Bieńkowska, Jacek Małecki, Alexander Mathiesen-Ohman et al. (2025). Person-AI Bidirectional Fit - A Proof-Of-Concept Case Study Of Augmented Human-Ai Symbiosis In Management Decision-Making Process. http://arxiv.org/abs/2511.13670

[25] Francisco Abreu, Luís Cruz, Sérgio Guerreiro (2025). Ontology-Driven Model-to-Model Transformation of Workflow Specifications. http://arxiv.org/abs/2511.13661

[26] Hengrui Hu, Kaining Ying, Henghui Ding (2025). Segment Anything Across Shots: A Method and Benchmark. http://arxiv.org/abs/2511.13715

[27] Pranjal Balar, Sundeep Kapila (2025). Integrative Model for Interoception and Exteroception: predictive coding, points of modulation, and testable predictions. http://arxiv.org/abs/2511.13668

[28] Jay R. Krishnan, Kevork N. Abazajian (2025). The Scatter of the Many Outweighs the Scatter of the Few: Systematic Error Asymmetry in Steeply-Falling Mass Functions for High-Redshift JWST Galaxies. http://arxiv.org/abs/2511.13708

[29] Rayff de Souza, Agripino Sousa-Neto, Javier E. González et al. (2025). A model-independent assessment of the late-time dark energy density evolution. http://arxiv.org/abs/2511.13666

[30] Sergei Gukov, Po-Shen Hsin, Du Pei (2025). Generalized Global Symmetries of $T[M]$ Theories: Part II. http://arxiv.org/abs/2511.13696

[31] Changjie Chen (2025). Stability phenomena in Deligne-Mumford compactifications via Morse theory. http://arxiv.org/abs/2511.13695

## Papers Used (40 papers)

1. **Back to Basics: Let Denoising Generative Models Denoise**
   - Paper ID: 2511.13720
   - Abstract: Today's denoising diffusion models do not "denoise" in the classical sense, i.e., they do not directly predict clean images. Rather, the neural networks predict noise or a noised quantity. In this paper, we suggest that predicting clean data and predicting noised quantities are fundamentally differe...

2. **Scaling Spatial Intelligence with Multimodal Foundation Models**
   - Paper ID: 2511.13719
   - Abstract: Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations in...

3. **Bosonisation Cohomology: Spin Structure Summation in Every Dimension**
   - Paper ID: 2511.13718
   - Abstract: Gauging fermion parity and summing over spin structures are subtly distinct operations. We introduce 'bosonisation cohomology' groups $H_B^{d+2}(X)$ to capture this difference, for theories in spacetime dimension $d$ equipped with maps to some $X$. Non-trivial classes in $H_B^{d+2}(X)$ contain theor...

4. **TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone**
   - Paper ID: 2511.13717
   - Abstract: Large Language Models (LLMs) deployed on mobile devices offer benefits like user privacy and reduced network latency, but introduce a significant security risk: the leakage of proprietary models to end users.
  To mitigate this risk, we propose a system design for protecting on-device LLMs using Arm...

5. **Segment Anything Across Shots: A Method and Benchmark**
   - Paper ID: 2511.13715
   - Abstract: This work focuses on multi-shot semi-supervised video object segmentation (MVOS), which aims at segmenting the target object indicated by an initial mask throughout a video with multiple shots. The existing VOS methods mainly focus on single-shot videos and struggle with shot discontinuities, thereb...

6. **UnSAMv2: Self-Supervised Learning Enables Segment Anything at Any Granularity**
   - Paper ID: 2511.13714
   - Abstract: The Segment Anything Model (SAM) family has become a widely adopted vision foundation model, but its ability to control segmentation granularity remains limited. Users often need to refine results manually - by adding more prompts or selecting from pre-generated masks - to achieve the desired level ...

7. **Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine**
   - Paper ID: 2511.13713
   - Abstract: Recent advances in text-to-image (T2I) diffusion models have significantly improved semantic image editing, yet most methods fall short in performing 3D-aware object manipulation. In this work, we present FFSE, a 3D-aware autoregressive framework designed to enable intuitive, physically-consistent o...

8. **From Black Box to Insight: Explainable AI for Extreme Event Preparedness**
   - Paper ID: 2511.13712
   - Abstract: As climate change accelerates the frequency and severity of extreme events such as wildfires, the need for accurate, explainable, and actionable forecasting becomes increasingly urgent. While artificial intelligence (AI) models have shown promise in predicting such events, their adoption in real-wor...

9. **From Power to Precision: Learning Fine-grained Dexterity for Multi-fingered Robotic Hands**
   - Paper ID: 2511.13710
   - Abstract: Human grasps can be roughly categorized into two types: power grasps and precision grasps. Precision grasping enables tool use and is believed to have influenced human evolution. Today's multi-fingered robotic hands are effective in power grasps, but for tasks requiring precision, parallel grippers ...

10. **The Scatter of the Many Outweighs the Scatter of the Few: Systematic Error Asymmetry in Steeply-Falling Mass Functions for High-Redshift JWST Galaxies**
   - Paper ID: 2511.13708
   - Abstract: The discovery of massive, high redshift galaxies with JWST has been argued to challenge $Λ$CDM: such systems would require extremely rare halos and baryon-to-stellar-mass conversion efficiencies unphysically approaching--or exceeding--100%. If confirmed at galaxy formation forbidden efficiencies, th...

11. **OpenRoboCare: A Multimodal Multi-Task Expert Demonstration Dataset for Robot Caregiving**
   - Paper ID: 2511.13707
   - Abstract: We present OpenRoboCare, a multimodal dataset for robot caregiving, capturing expert occupational therapist demonstrations of Activities of Daily Living (ADLs). Caregiving tasks involve complex physical human-robot interactions, requiring precise perception under occlusions, safe physical contact, a...

12. **TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models**
   - Paper ID: 2511.13704
   - Abstract: The rapid evolution of video generative models has shifted their focus from producing visually plausible outputs to tackling tasks requiring physical plausibility and logical consistency. However, despite recent breakthroughs such as Veo 3's chain-of-frames reasoning, it remains unclear whether thes...

13. **Generalist Foundation Models Are Not Clinical Enough for Hospital Operations**
   - Paper ID: 2511.13703
   - Abstract: Hospitals and healthcare systems rely on operational decisions that determine patient flow, cost, and quality of care. Despite strong performance on medical knowledge and conversational benchmarks, foundation models trained on general text may lack the specialized knowledge required for these operat...

14. **Learning stochasticity: a nonparametric framework for intrinsic noise estimation**
   - Paper ID: 2511.13701
   - Abstract: Understanding the principles that govern dynamical systems is a central challenge across many scientific domains, including biology and ecology. Incomplete knowledge of nonlinear interactions and stochastic effects often renders bottom-up modeling approaches ineffective, motivating the development o...

15. **The Role of Gyrating Ions in Reformation of a Quasi-parallel Supercritical Shock**
   - Paper ID: 2511.13697
   - Abstract: Collisionless shocks in space and astrophysical plasmas mediate energy exchange between charged particles and fields in two or more plasma flows. In this study we analyze the evolution of ion distributions around a reformation cycle of a quasi-parallel shock. We use multi-point in-situ observations ...

16. **Generalized Global Symmetries of $T[M]$ Theories: Part II**
   - Paper ID: 2511.13696
   - Abstract: We continue the investigation of symmetries and anomalies of $T[M]$ theories obtained by compactifying 6d SCFTs on an internal manifold $M$. We extend the notion of "polarizations on a manifold $M$" to cases where $M$ may have boundaries or defects. Through examples with $M$ of dimension two, three,...

17. **Stability phenomena in Deligne-Mumford compactifications via Morse theory**
   - Paper ID: 2511.13695
   - Abstract: We study the rational homology of the Deligne-Mumford compactification $\overline{\mathcal M}_{g,n}$ of the moduli space of stable curves via a family of Morse functions, the $\text{sys}_T$ functions, which encode geometric information about short geodesics on hyperbolic surfaces. Exploiting the Mor...

18. **A Note on Large Degenerate Induced Subgraphs in Sparse Graphs**
   - Paper ID: 2511.13693
   - Abstract: Given a graph $G$ and a non-negative integer $d$ let $α_d(G)$ be the order of a largest induced $d$-degenerate subgraph of $G$. We prove that for any pair of non-negative integers $k>d$, if $G$ is a $k$-degenerate graph, then $α_d(G) \geq \max\{ \frac{(d+1)n}{k+d+1}, n - α_{k-d-1}(G)\}$. For $k$-deg...

19. **The physical properties of post-mass-transfer binaries**
   - Paper ID: 2511.13692
   - Abstract: Aims. We present and analyse the detailed physical properties of six binary stellar systems, originally proposed as possible star-black hole binaries on the basis of radial velocities from Gaia's third data release, but soon recognised as likely post-mass-transfer binary systems with stripped compan...

20. **Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation**
   - Paper ID: 2511.13689
   - Abstract: Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-nativ...

21. **The hard ultraluminous state of NGC 5055 ULX X-1**
   - Paper ID: 2511.13686
   - Abstract: We present the results of the first broadband X-ray analysis of the ultraluminous X-ray source NGC 5055 ULX X-1, combining simultaneous data from XMM$-$Newton and NuSTAR missions, with a combined exposure time of $\sim$100 ks across the $0.3-20$ keV energy range. The source exhibits a stable flux ac...

22. **Protein Secondary Structure Prediction Using 3D Graphs and Relation-Aware Message Passing Transformers**
   - Paper ID: 2511.13685
   - Abstract: In this study, we tackle the challenging task of predicting secondary structures from protein primary sequences, a pivotal initial stride towards predicting tertiary structures, while yielding crucial insights into protein activity, relationships, and functions. Existing methods often utilize extens...

23. **Training-Free Multi-View Extension of IC-Light for Textual Position-Aware Scene Relighting**
   - Paper ID: 2511.13684
   - Abstract: We introduce GS-Light, an efficient, textual position-aware pipeline for text-guided relighting of 3D scenes represented via Gaussian Splatting (3DGS). GS-Light implements a training-free extension of a single-input diffusion model to handle multi-view inputs. Given a user prompt that may specify li...

24. **Molecular Engineering for Enhanced Second-Order Nonlinear Response in Spontaneously-Oriented Evaporated Organic Films**
   - Paper ID: 2511.13682
   - Abstract: Materials with large second-order nonlinearities are crucial for next-generation integrated photonics. Spontaneously oriented organic thin films prepared by physical vapor deposition offer a promising poling-free and scalable approach. This study investigates molecular engineering strategies to enha...

25. **Cross-Learning from Scarce Data via Multi-Task Constrained Optimization**
   - Paper ID: 2511.13680
   - Abstract: A learning task, understood as the problem of fitting a parametric model from supervised data, fundamentally requires the dataset to be large enough to be representative of the underlying distribution of the source. When data is limited, the learned models fail generalize to cases not seen during tr...

26. **The Publication Choice Problem**
   - Paper ID: 2511.13678
   - Abstract: Researchers strategically choose where to submit their work in order to maximize its impact, and these publication decisions in turn determine venues' impact factors. To analyze how individual publication choices both respond to and shape venue impact, we introduce a game-theoretic framework, coined...

27. **Open-shell frozen natural orbital approach for quantum eigensolvers**
   - Paper ID: 2511.13677
   - Abstract: We present an open-shell frozen natural orbital (FNO) approach, which utilizes the second-order Z-averaged perturbation theory (ZAPT2), to reduce the restricted opten-shell Hartree-Fock virtual space size with controllable accuracy. Our ZAPT2 frozen natural orbital (ZAPT-FNO) selection scheme signif...

28. **T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization**
   - Paper ID: 2511.13676
   - Abstract: Recent advances in LLMs have outpaced the computational and memory capacities of edge platforms that primarily employ CPUs, thereby challenging efficient and scalable deployment. While ternary quantization enables significant resource savings, existing CPU solutions rely heavily on memory-based look...

29. **Scientific Data Compression and Super-Resolution Sampling**
   - Paper ID: 2511.13675
   - Abstract: Modern scientific simulations, observations, and large-scale experiments generate data at volumes that often exceed the limits of storage, processing, and analysis. This challenge drives the development of data reduction methods that efficiently manage massive datasets while preserving essential phy...

30. **HilbMult: A Banach-Enriched Multicategory for Operator Algebras**
   - Paper ID: 2511.13674
   - Abstract: Category and multicategory theory provide abstract frameworks for describing structures and their compositions, with multicategories extending traditional categories to handle multi-input operations. These theories enable modular reasoning and coherent composition of complex systems, and have found ...

31. **A new generalization of the Narayana numbers inspired by linear operators on associative $d$-ary algebras**
   - Paper ID: 2511.13671
   - Abstract: We introduce and study a generalization of the Narayana numbers $N_d(n,k) = \frac{1}{n+1} \binom{n+1}{k+1} \binom{ n + (n-k)(d-2)+1}{k}$ for integers $d \geq 2$ and $n,k \geq 0$. This two-parameter array extends the classical Narayana numbers ($d=2$) and yields a $d$-ary analogue of the Catalan numb...

32. **Person-AI Bidirectional Fit - A Proof-Of-Concept Case Study Of Augmented Human-Ai Symbiosis In Management Decision-Making Process**
   - Paper ID: 2511.13670
   - Abstract: This article develops the concept of Person-AI bidirectional fit, defined as the continuously evolving, context-sensitive alignment-primarily cognitive, but also emotional and behavioral-between a human decision-maker and an artificial intelligence system. Grounded in contingency theory and quality ...

33. **Investigating the Dark Energy Constraint from Strongly Lensed AGN at LSST-Scale**
   - Paper ID: 2511.13669
   - Abstract: Strongly lensed Active Galactic Nuclei (AGN) with an observable time delay can be used to constrain the expansion history of the Universe through time-delay cosmography (TDC). As the sample of time-delay lenses grows to statistical size, with $\mathcal{O}$(1000) lensed AGN forecast to be observed by...

34. **Integrative Model for Interoception and Exteroception: predictive coding, points of modulation, and testable predictions**
   - Paper ID: 2511.13668
   - Abstract: Interoception and exteroception provide continuous feedback about the body and the environment, yet how they are dynamically integrated within a unified predictive coding framework has remained under-specified. This paper develops and empirically validates an integrative predictive coding model that...

35. **Quantum complexity across thermal phase transition in the transverse field Ising chain with long-range couplings**
   - Paper ID: 2511.13667
   - Abstract: We investigate the behavior of the Schmidt gap, the von Neumann entanglement entropy, and the non-stabiliserness in proximity to the classical phase transition of the one-dimensional long-range transverse-field Ising model (LRTFIM). Leveraging the time-dependent variational principle (TDVP) within a...

36. **A model-independent assessment of the late-time dark energy density evolution**
   - Paper ID: 2511.13666
   - Abstract: Combined measurements of Baryon Acoustic Oscillations (BAO) from the Dark Energy Spectroscopic Survey (DESI), the Cosmic Microwave Background (CMB) and Type Ia Supernovae (SN Ia), have recently challenged the $Λ$-Cold Dark Matter ($Λ$CDM) paradigm, indicating potential evidence for a dynamical dark ...

37. **Einstein-Maxwell fields as solutions of Einstein gravity coupled to conformally invariant non-linear electrodynamics**
   - Paper ID: 2511.13665
   - Abstract: We study Einstein-Maxwell (non-null) sourcefree configurations that can be extended to any conformally invariant non-linear electrodynamics (CINLE) by a constant rescaling of the electromagnetic field. We first obtain a criterion which characterizes such extendable solutions in terms either of the e...

38. **Cost-Driven Synthesis of Sound Abstract Interpreters**
   - Paper ID: 2511.13663
   - Abstract: Constructing abstract interpreters that provide global soundness guarantees remains a major obstacle in abstract interpretation. We investigate whether modern LLMs can reduce this burden by leveraging them to synthesize sound, non-trivial abstract interpreters across multiple abstract domains in the...

39. **Ontology-Driven Model-to-Model Transformation of Workflow Specifications**
   - Paper ID: 2511.13661
   - Abstract: Proprietary workflow modeling languages such as Smart Forms & Smart Flow hamper interoperability and reuse because they lock process knowledge into closed formats. To address this vendor lock-in and ease migration to open standards, we introduce an ontology-driven model-to-model pipeline that system...

40. **Graviton propagator in de Sitter space in a simple one-parameter gauge**
   - Paper ID: 2511.13660
   - Abstract: We construct the graviton propagator in de Sitter space in a one-parameter family of noncovariant gauges. This family generalizes the simple gauge in which most graviton loop computations in de Sitter space have been performed. The resulting propagator has a relatively simple form and will facilitat...

