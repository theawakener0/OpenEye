# Methodology

## OpenEye: The New Revolution of SLMs

The methodology behind OpenEye follows a single guiding principle: “intelligence should expand through depth, not
size.” Inspired by the research direction proposed in Samsung’s “Less is More: Recursive Reasoning with Tiny Networks” (2025), OpenEye adopts a minimalistic yet recursive design philosophy that redefines what small-scale, on-device intelligence can achieve.

Our development framework combines open-source modularity, recursive optimization, and small model autonomy to
create a wearable system that is both lightweight and deeply intelligent. By leveraging these ideas, OpenEye seeks
to achieve the efficiency and adaptability of Tiny Recursive Models (TRMs) within the context of Small Language
Models (SLMs) for real-world interaction.

1. **Design Principles**:
   - **Modularity**: The system is designed to be modular, allowing users to easily add or remove components and functionalities.
   - **Offline Functionality**: Emphasis on offline capabilities to ensure privacy and reduce dependency on cloud services.
   - **User-Centric Design**: Focus on creating an intuitive user interface that enhances user experience.
   - **Scalability**: The architecture is built to support future expansions and integrations with other technologies.

2. **Development Process**:
   - **Open-Source Collaboration**: Leveraging community contributions to enhance the system's features and capabilities.
   - **Agile Development**: Implementing an agile development process to allow for iterative improvements and rapid prototyping.
   - **Hardware-Software Integration**: Ensuring seamless integration between hardware components and software functionalities.
   - **Optimization for SLMs**: Tailoring the system to optimize performance for Small Language Models (SLMs) to ensure efficiency and adaptability.

3. **Evaluation Metrics**:
   - **Performance Testing**: Assessing the system's responsiveness, latency, and overall performance.
   - **User Feedback**: Collecting and analyzing user feedback to identify areas for improvement.
   - **Security and Privacy**: Evaluating the system's ability to protect user data and ensure privacy.

4. **Integrating “Less is More” in OpenEye’s Evolution**:
    Samsung’s “Less is More” research demonstrated that tiny recursive networks can outperform massive language
    models on complex reasoning tasks. This finding redefines how intelligence can emerge from small systems.

    OpenEye builds upon this insight by adopting similar principles in the wearable domain:
    - **Recursive Inference**: Each SLM in OpenEye recursively refines its output based on internal feedback.
    - **Tiny Model Optimization**: Instead of increasing model size, we enhance performance through recursion and
    compression.
    - **Energy-Aware Intelligence**: Smaller models mean lower power consumption, essential for long-term wearable
    autonomy.

    Thus, Less is More is not only a reference but a methodological foundation for OpenEye’s design, guiding us
    toward a form of intelligence that grows in clarity, not in size.


Before we start in showing the methodology, it's important to understand the idea behind OpenEye and how it differs from conventional smart glasses. Unlike traditional systems that rely heavily on cloud infrastructure, OpenEye is designed to operate independently, providing users with a more secure and efficient experience.

And how we achieve this is through our novel computational architecture that enables Small Language Models (SLMs) to function effectively on-device. This approach minimizes latency, enhances privacy, and allows for seamless functionality without the need for constant internet connectivity.

## Beyond SLMs

Why we are using SLMs instead of LLMs? The answer is simple: efficiency and contextual adaptability. SLMs are optimized to perform well on smaller devices, making them ideal for wearable technology like smart glasses. By focusing on SLMs, we can ensure that OpenEye delivers high performance while maintaining a compact form factor.

But by using SLMs, we will encounter some challenges, such as limited computational resources and the need for effective model optimization. To address these challenges, our methodology includes strategies for model compression, efficient data processing, and adaptive learning techniques.

We believe that by leveraging SLMs, we can create a more responsive and personalized user experience, allowing the smart glasses to adapt to individual user needs and preferences.

## Introduction to our method of making OpenEye

During the earliest phase of building our framework, we experimented with existing open-source components to speed up development. Very quickly, it became evident that these components created inherent performance ceilings. Their abstraction layers, memory behavior, and fixed execution pipelines made deep optimization, especially on constrained edge hardware, nearly impossible. Optimizing someone else’s architecture is fundamentally inefficient, so we decided to build the entire system from scratch to fully control concurrency, memory flow, and hardware interaction.

Language selection became the next critical decision point.

Python’s ecosystem is strong but fundamentally unsuitable for real-time, on-device workloads. C++ offered raw performance but would restrict extensibility and complicate plugin development for an open-source community. Golang provided the optimal balance: native speed, predictable memory behavior, built-in concurrency, and an approachable toolchain. For a pluggable, hardware-aware AI framework, Go is both practical and strategically correct.

With the language chosen, we moved on to validating how the framework would interact with Small Language Models (SLMs). Our first prototype used llama.cpp as a lightweight model runner. We ran its REST API, then built a minimal Go wrapper: a TCP server connected to the model runner and a TCP client that transmitted prompts. The setup was intentionally simple. It allowed us to observe raw dataflow, concurrency behavior, latency patterns, and scheduling overhead without external interference.
The test worked as intended. Qwen3-0.6B responded through our pipeline with low latency. This proved our communication layer was solid and provided a baseline for further architectural development.

We then expanded the prototype into a full SLM framework still inspired by the simplicity of the initial test but now enriched with modular subsystems, optimized neural execution paths, pluggable extensions, and hardware adaptive scheduling.

This design direction is strongly reinforced by contemporary research:

Neuromorphic Edge AI (Ferreira et al., 2025).
Their comparative review of deep and spiking neural networks demonstrates that edge-native intelligence demands architectural efficiency, event-driven computation, and memory-aware execution. This confirms our decision to avoid heavy general purpose frameworks and to engineer the runtime ourselves for deterministic, low power behavior.

Embedded Neuromorphic Platforms (ColibriES – Rutishauser et al., 2023).
ColibriES integrates neuromorphic accelerators and traditional neural hardware into a unified embedded control system achieving milliwatt level operation with real-time performance. This aligns directly with our hardware-aware framework philosophy and provided a reference point for designing tightly coupled hardware runtime interaction.

DeepSeek OCR.
DeepSeek’s OCR and long-document reasoning system demonstrates how lightweight, highly optimized perception modules can outperform bulky multimodal stacks. Their pipeline validates our approach of integrating efficient perception components into the framework rather than relying on external, monolithic systems.

Edge AI deployment (Neurocomputing Survey, 2025).
This survey highlights a consistent trend: effective edge AI requires co-design of the model runner, inference engine, and hardware abstraction layer, precisely the architectural choice behind OpenEye. Their findings directly support our decision to build every layer ourselves and to keep the framework lightweight, deterministic, and deeply hardware-aware.

Together, these works reinforce that our approach is correct: small models, deeply optimized runtimes, controlled memory flow, and event driven computation consistently outperform large, generalized frameworks when deployed on real edge hardware.

What began as a minimal TCP-based experiment has evolved into a complete SLM framework with a modular architecture, pluggable components, optimized scheduling, and the capacity to integrate both classic neural models and future neuromorphic designs. By aligning the framework with both modern research and real hardware constraints, OpenEye is engineered for speed, extensibility, and true on-device intelligence.

### What is our framework architecture?

Our architecture is built around a core objective: extract the maximum possible performance from Small Language Models (SLMs) by optimizing context flow, memory, and control structures. SLMs become significantly more capable when given a tightly managed context—one that provides purpose, continuity, and situational awareness without overwhelming the model with unnecessary tokens. For this reason, our context system is divided into two tightly optimized components:
(1) System + Prompt Architecture and (2) Memory Architecture.

1. System Message & Prompt Architecture

Designing the system message and prompt layer required acknowledging a fundamental limitation identified in modern edge-AI literature: small models degrade quickly when overloaded with long or unstructured context. Studies in neuromorphic and edge-AI systems—such as Ferreira et al. (2025) and the broader Neurocomputing 2025 survey—show that efficiency, token discipline, and controlled input flow are critical for stable reasoning on constrained hardware.

Based on this, we imposed strict constraints:

The system message is capped below 100 tokens, ensuring the model receives clear behavioral rules without triggering hallucination or drift.

The global context format is optimized to remain minimal, deterministic, and impossible to bloat.

The system message becomes a configurable component, but bounded by mandatory constraints to avoid model instability.

We also implemented a dedicated context builder function. This function constructs the final prompt by merging the system message and the user’s input based on predefined rules, ensuring consistent behavior and preventing over-expansion. By enforcing token austerity and structured prompt assembly, we eliminate the most common failure modes of SLMs on edge hardware.

2. Memory Architecture: The Evolution to Omem

The memory subsystem required deeper architectural intervention. Our initial experiments with simple sliding windows (Legacy) and direct vector-store implementations (standard RAG) revealed significant limitations for SLMs: they either lacked long-term retention or suffered from context fragmentation.

To address this, we developed **Omem (OpenEye Memory)**, a native Go-based memory system designed specifically for the constraints of edge-based SLMs. This architecture evolved through two distinct phases:

**Phase A: Porting Mem0 to Native Go**
We initially looked to the Python-based `mem0` library for its vector-store capabilities. However, integrating a Python runtime on edge devices introduced unacceptable latency and overhead. We re-architected the system in Go, replacing the Python logic with a native implementation that utilizes **DuckDB** for vectorized storage and standard SQL relationships for graph operations. This removed the cgo/execution overhead while maintaining the core vector-search capabilities.

**Phase B: Omem (Adaptive Cognitive Architecture)**
While the port provided speed, standard vector retrieval proved insufficient for maintaining global coherence. Omem introduces a four-pillar architecture to solve this:

*   **Atomic Encoding:** Instead of storing raw user input, Omem uses an LLM pass to de-noise and resolve coreferences (e.g., changing "he did it" to "John completed the task") before storage. This ensures that retrieved facts are standalone and intelligible.
*   **Multi-View Indexing:** To overcome the limitations of semantic-only search, Omem indexes data in three simultaneous views: **Semantic** (vector embeddings for meaning), **Lexical** (BM25 for exact keyword matching), and **Symbolic** (graph relationships for entity connection).
*   **Adaptive Retrieval:** Unlike static Top-K retrievers, Omem calculates a **Complexity Score** for each query. Simple queries trigger a narrow, precise search, while complex reasoning tasks expand the search radius to gather broader context.
*   **Rolling Summarization:** To maintain global context without token bloat, Omem maintains a persistent "user biography" summary. This summary is updated incrementally and injected into every prompt, ensuring the model always possesses a baseline understanding of the user.

3. Context Window Optimization (Sliding Window)

Once system messages and memory were stable, we applied a sliding-window mechanism to the model’s context window.
This technique ensures:

The model always receives the most relevant recent information.

Total context length remains constrained regardless of conversation history.

Token load is predictable—avoiding instability and hallucination.

Sliding windows are widely supported by edge-AI literature, especially in models operating under strict hardware constraints. This aligns with findings from deep vs. spiking neural network comparisons (Ferreira et al., 2025) and edge-AI runtime studies, which emphasize temporal relevance over raw context length.

**Summary**

By combining:

- Strict token discipline for system messages

- Deterministic, compressed memory retrieval (DuckDB + summarization)

- Unified context orchestration

- A configurable sliding-window mechanism

we engineered a context system designed specifically for SLM stability and maximum performance.

This architecture directly follows insights from neuromorphic computing, efficient OCR pipelines (DeepSeek), and modern edge-AI research, all of which validate the principles we applied: minimalism, determinism, and hardware-aware optimization.

After building the context architecture, it became clear that this alone was not sufficient to fully unlock the capabilities of our SLM. While Omem solved the problem of *personal* and *episodic* memory (remembering the user), the model still lacked access to deep *external* knowledge (manuals, encyclopedic data, technical specs) that exceeds its training data.

The next logical enhancement was *External Retrieval-Augmented Generation (RAG)*, a mechanism distinct from Omem that acts as an infinite external library for the SLM.

**Differentiation: Omem vs. External RAG**
It is crucial to distinguish between these two retrieval systems in our architecture, as they serve fundamentally different cognitive functions:

*   **Omem (Internal Memory):** Handles *who the user is*. It retrieves past conversations, preferences, and biographical facts. It uses a graph-based approach to maintain continuity.
*   **External RAG (Knowledge Base):** Handles *what the world is*. It retrieves static documents, API references, or uploaded files. It uses a pure semantic search approach.

The OpenEye pipeline orchestrates both systems in parallel, merging their outputs into a single coherent context.

**Retrieval-Augmented Generation (RAG) Integration**

External RAG gives the model access to information far beyond what it can store in context or internal parameters. Instead of forcing the SLM to memorize or repeatedly reprocess data, we externalize long-term knowledge into a structured retrieval layer. This allows the SLM to:

- Recall older interactions beyond the sliding window

- Access large knowledge bases in real time

- Generate responses with higher factual accuracy

- Avoid hallucinations by grounding outputs in retrieved data

- Maintain continuity across long sessions without storing large prompts

This design is strongly supported by modern lightweight retrieval research. Both DeepSeek’s OCR system and neuromorphic memory studies (e.g., Ferreira et al., 2025) emphasize the importance of external structured memory for stable reasoning under constrained computation. Likewise, the Neurocomputing 2025 edge-AI survey highlights that externalized, relevance-filtered memory is essential for small models, especially when operating on-device.

**RAG Pipeline Design**

Our RAG system is designed to be compact, hardware-aware, and consistent with the philosophy of our framework. It follows four stages:

Storage Layer (DuckDB)
We leverage DuckDB’s vectorized execution and columnar storage to efficiently index documents, embeddings, interaction logs, structured facts, and compressed memory.

Embedding Generation (SLM-compatible)
Instead of using heavyweight embedding models, we generate embeddings using:

an extremely lightweight transformer encoder, or the SLM itself through a projection layer.
This avoids dependency on GPU-heavy embedding pipelines.

Retriever (Top-K, MMR, and Semantic Filtering)
Before injecting retrieved data into the model, we run:

- top-K semantic retrieval

- Maximal Marginal Relevance (MMR) de-duplication

- context-fit scoring

- strict token budgeting

This prevents irrelevant memory injection and eliminates retrieval-driven hallucination.

RAG Context Builder
Retrieved information is not passed raw.

Instead, we apply:

compression algorithms (similar to our memory architecture)

summarization to a fixed token budget

deterministic formatting
The final output becomes a clean, compact “knowledge block" injected into the global context.

RAG Advantages for SLM Performance

With RAG, the SLM gains abilities previously impossible:

Infinite Memory:
The model can recall events, instructions, topics, or knowledge from any point in time, not just the recent sliding window.

High Factual Stability:
Retrieved facts act as grounding anchors, reducing hallucination—a behavior also emphasized in DeepSeek’s document-grounded pipeline.

Dynamic Knowledge Injection:
The model becomes capable of updating itself in real time without re-training.
This mirrors trends in modern neuromorphic and edge-AI systems where external memory modules provide adaptive intelligence.

Lower Token Load:
Instead of storing long conversations or documents in context, we store them externally and retrieve on demand.

Summary: RAG as the Final Layer of Intelligence

RAG transforms the SLM from a reactive model into a system with persistent, dynamic, and scalable intelligence.
By combining:

- our optimized context architecture

- deterministic memory compression

- hardware-aware scheduling and RAG’s infinite external memory

the SLM becomes substantially more accurate, more powerful, and more stable without increasing model size or hardware requirements.

This external-memory strategy is consistent with modern research across neuromorphic computing, OCR-grounded systems, and efficient edge-AI design. It also positions the framework as a future-proof platform capable of supporting advanced SLM reasoning on any device.

Building the context was not enough to achieve our idea of improving the SLMs model, so what we did next was to make the framework usable to the developers, as it was just some lines of code that can not be used as a framework, so what we did next is to build the pipeline that will organise and make the framework easy to use.

#### How the pipeline works

To understand the framework’s methodology, we must first clarify the role of the pipeline. The pipeline functions as the primary interface layer between developers and the internal components of the framework. In practice, it serves as the framework’s Software Development Kit (SDK), a stable, high-level interaction point that abstracts the underlying complexity and exposes a clean, extensible API for integrating SLMs into applications.

This interface is critical for two reasons.
First, it ensures the framework is accessible to developers who want to extend, modify, or embed it in external systems. Second, it defines the architectural boundary that enables the project to remain open-source, modular, and community-driven—an essential goal of the entire design.

The internal structure of the pipeline is intentionally balanced. It is neither overengineered nor oversimplified; instead, it is built to perform its function consistently, predictably, and efficiently.
The pipeline’s design builds directly on our earliest prototype: a minimal TCP-based interaction between the model runner and a wrapper layer. Over time, we expanded this prototype into a fully integrated interface that connects all components of the framework into a single, cohesive execution path.

With the introduction of the context architecture (system messages, prompt engine, memory module, and RAG subsystem), the pipeline now orchestrates these elements in real time. When a developer initializes the pipeline, several processes are triggered internally:

Context Initialization
The pipeline constructs the initial context using the system message, configuration rules, and sliding-window constraints. Crucially, it triggers the **Parallel Context Assembly**:
*   **Omem Hook:** Asynchronously queries the Omem engine to retrieve the "User Biography" (Rolling Summary) and relevant past episodic memories (Adaptive Retrieval).
*   **RAG Retriever:** Simultaneously queries the external vector store for knowledge documents relevant to the current query.
*   **Vector Cache:** Fetches the most recent immediate conversation turns.

These elements are then merged by the context builder into a single, deterministic, token-optimized prompt.

Component Activation
The pipeline loads and configures the modules associated with the current model, including:

the SLM runner interface

the memory subsystem (DuckDB engine, summarization layers)

the Omem adapter (Long-term memory integration)

the retrieval layer (for RAG)

any developer defined extensions or plugins

Model Binding
The pipeline binds the prepared context to the selected SLM. This includes token budgeting, embedding alignment, and any additional reinforcement rules derived from the configuration.

Runtime Execution Path Setup
The data flow path is established:

incoming user input → pre-processing → **Parallel Retrieval (Omem + RAG)** → context reconstruction → model inference → post-processing and grounding → **Async Memory Consolidation (Omem learning)**.

From this moment onward, the pipeline acts as the central conductor. Every query from a developer’s application is passed through the same controlled pathway—with strict optimization guarantees—ensuring the SLM remains stable, accurate, and hallucination-resistant.

By encapsulating all internal modules behind a single, unified pipeline, the framework achieves two objectives simultaneously:
(1) predictable, high-performance operation for every SLM, and
(2) a simple, developer-friendly interface that abstracts away complexity without sacrificing control.

### Native Inference Engine

OpenEye implements a custom native inference layer using Go-based llama.cpp CGo bindings, eliminating HTTP/REST overhead present in client-server architectures. This zero-copy approach enables direct memory management and hardware-aware execution optimized for ARM edge devices.

**Key Components:**
- **Direct Model Loading**: Load GGUF models directly without serialization overhead, enabling sub-millisecond initialization
- **Hardware-Aware Threading**: Configurable thread pools for inference and batch processing, with automatic CPU core detection
- **ARM NEON Optimization**: Vector operation paths optimized for Raspberry Pi and ARM edge devices
- **Multi-Modal Support**: Vision model support via mmproj files for integrated camera processing
- **Prompt Caching**: Stores last prompt's token sequence to reuse KV cache prefix, dramatically reducing time-to-first-token (TTFT) on edge devices by eliminating redundant prompt re-evaluation
- **Sampler Reuse**: Avoids per-request allocation of CGo sampler chains; chains are rebuilt only when sampling parameters change between requests

The native adapter implements the same `runtime.Adapter` interface as the HTTP backend, allowing seamless switching between local native inference and remote server-based inference via configuration changes.

### Inference Optimization Stack

Our optimization approach draws inspiration from Dynamic Memory Compression (NVIDIA, 2024), which demonstrated that adaptive memory reduction can significantly improve inference efficiency without sacrificing model quality. While DMC operates at the model architecture level—training models to adaptively compress KV cache through learned accumulation decisions—we apply similar principles at the inference runtime level through configurable quantization and caching strategies. This runtime-level approach is essential for edge deployment where users cannot modify or retrain model architectures.

**The Eight Optimization Techniques:**

1. **KV Cache Quantization** (DMC-inspired): Configurable precision levels (f16, q8_0, q4_0) for KV cache storage. At q4_0, achieves approximately 75% KV memory reduction while maintaining acceptable quality for edge deployment. This directly applies DMC's memory efficiency principles without requiring model retraining.

2. **Speculative Decoding**: A smaller "draft" model (e.g., Gemma-3-270M, 270M parameters) generates candidate tokens that the target model (e.g., LFM2.5-1.2B, 1.2B parameters) verifies in batch. When verification passes, multiple tokens are accepted per decode step, yielding 1.5-2.5x throughput improvement on CPU-bound edge devices.

3. **Flash Attention**: Memory-efficient attention computation that reduces memory pressure during long-context inference, enabling larger effective context windows within the same memory budget.

4. **Prompt Caching**: Reuses KV cache prefix across requests with identical system prompts and conversation history, eliminating redundant computation and reducing TTFT for subsequent turns in a conversation.

5. **Context Shifting**: Automatic sliding window mechanism that discards the oldest context when KV cache fills, enabling unbounded conversation length without generation failure or context overflow errors.

6. **Greedy Fast-Path**: Temperature=0.2 triggers an optimized sampling path with reduced computational overhead for deterministic generation scenarios, improving throughput for factual queries.

7. **Stream Chunking**: Configurable token buffering (default: 3 tokens) before emitting to stream callback, enabling smoother word-level streaming rather than character-by-character output, improving perceived responsiveness.

8. **Parallel Context Assembly**: Four concurrent goroutines handle context building operations (summarization, vector search, RAG retrieval, Omem context assembly), hiding I/O latency behind computation and reducing overall request latency.

These optimizations can be selectively enabled or disabled via configuration, allowing users to tune the performance-quality trade-off for their specific hardware constraints and use cases.

### OpenEye Smart Glasses – Hardware Architecture & Modular Design

The hardware architecture of the OpenEye smart glasses is built around a core principle that aligns with the OpenEye framework itself: pluggability and extensibility at every layer. Just as the software framework exposes pluggable modules, hardware must remain modular, replaceable, and upgradable without requiring a redesign of the entire system. This approach enables incremental improvement, supports experimentation with different compute modules, and ensures long-term sustainability of the platform.

Consistent with the OpenEye architectural goals

and the principles of tiny-model, hardware-aware intelligence demonstrated in TRM research

, the smart glasses hardware evolves through three major phases. Each phase increases the degree of integration, energy efficiency, local compute capability, and hardware modularity.

1. Phase One — Prototype Platform (Completed)
Goal: Functional proof-of-concept with maximum hardware flexibility.

Phase One adopts a highly modular, rapid-prototyping approach, allowing each hardware element to be swapped or replaced without system-level redesign. This phase demonstrates the core interaction loop while enabling iteration at minimal cost.

Hardware modules (fully pluggable in Phase 1)

Compute Module: ESP32-CAM (ESP32 + OV2640 camera)
Pluggable: the ESP32 board can be swapped for other Wi-Fi MCUs or camera modules without affecting the display, audio, or power systems.

Display Module: Simple OLED screen
Pluggable: interchangeable I²C/SPI OLEDs.

Audio Module: Bluetooth earbuds (AirPods)
Pluggable: any BLE headset can be paired.

Mobile-side Compute Module:
The phone acts as an external compute plug-in (server on Termux), effectively serving as a detachable accelerator.

Power Module: Off-the-shelf Li-Po and USB chargers
Pluggable: no dependency on custom PMICs.

System operation

Camera → phone inference → ESP32/OLED output → BLE audio.
Because each module is detached and replaceable, this phase validates the entire pipeline while keeping costs low and allowing modifications at every step.

2. Phase Two — Integrated Edge Compute Platform (Raspberry Pi 5, 8 GB)
Goal: Migrate compute on-device while preserving modularity and enabling richer plug-in extensions.

Phase Two replaces the external phone with an on-device SBC capable of running optimized SLMs locally. This step is essential for reducing dependency on cloud or external devices—consistent with OpenEye’s on-device intelligence objectives.

Modular System Block (Plug-in Architecture)

To support extensibility, the hardware is structured as a set of swappable blocks, each with a defined electrical and software interface:

Compute Block (Plug-in Compute Module)

Raspberry Pi 5 (8 GB)

Connected via a standard connector interface (USB-C, ribbon, or edge connector)

Future modules (RISC-V boards, ARM SoCs, NPUs) can replace the Pi without altering cameras, audio, or PMIC.

Sensor Block (Camera/IMU Subsystem)

CSI/MIPI camera modules remain fully swappable

IMU optional plug-in for head/gesture tracking

Display Block

Micro-OLED or LCOS display driven via DSI or bridge IC

Visual subsystem remains separate from compute, allowing future upgrades without altering compute block

Audio Block

BLE earbuds or bone-conduction drivers

Maintains full interchangeability

Power Block

Battery, PMIC, and regulators acting as a separate module

Compute and sensors depend on it but do not integrate with it directly, preserving swappability

Storage Block

NVMe or eMMC on an M.2/PCIe adapter

Can be exchanged depending on model sizes or RAG database needs

The structured modularity reflects the OpenEye design ideology: every hardware subsystem must function as a replaceable plug-in, much like software modules in the OpenEye framework.

Justification supported by research

Tiny models and efficient reasoning approaches benefit from on-device acceleration.

Edge computing research (Neurocomputing 2025 survey) supports the need for low-latency, modular hardware for deployed AI.

The neuromorphic modularity seen in ColibriES (2023) similarly motivates separating sensor, compute, and power blocks into independent, replaceable modules.

3. Phase Three — Final Multi-Layer PCB (Production Level)
Goal: A fully integrated, ultra-compact PCB designed for mass production, but still fully pluggable and extensible.

Phase Three consolidates the system into a custom PCB while preserving the pluggable ethos. This is achieved through a multi-board modular stack rather than a monolithic board.

Final Architecture: Pluggable Embedded System
A. Primary Compute Module (Detachable Processor Board)

A SOM (System-on-Module) or edge-mounted compute card

Connected through a high-density board-to-board connector

Supports multiple future CPU/NPU options without redesigning the sensor or power systems

This allows future upgrades: e.g., swapping an ARM compute board for a RISC-V module optimized for TRM-style reasoning.

B. Sensor & Camera Module (Independent Plug-in)

Camera is not soldered to the compute board

Uses a flex cable or micro-board connector

Enables improvements to optics, sensor resolution, or event-camera modules without touching other systems

This mimics the modular neuromorphic sensor-block seen in ColibriES.

(via neuromorphic camera support)

C. Display Optics Module (Swappable Display Engine)

Micro-OLED/LCoS projector module

Pluggable connectors for fast prototyping of different display technologies

Allows individual experimentation with brightness, resolution, and form factor

D. Power & PMIC Module (Isolated Energy Subsystem)

Power-handling electronics are isolated into a separate board section

Supports battery upgrades, PMIC swapping, and tunable power profiles

Protects compute board from redesign when battery chemistry or capacity changes

E. I/O & Connectivity Expansion Module

USB-C, Wi-Fi/Bluetooth radio, and debug interface placed on a detachable I/O daughterboard

Enables regulatory, RF, and antenna control without redesigning compute core

Benefits of this modular PCB design

Extensibility: future hardware improvements require replacing only one board, not the entire glasses.

Repairability: broken modules can be swapped by users or developers.

Developer-friendly: allows 3rd-party compute modules, custom sensor modules, or extended radio modules.

Software–hardware co-design: aligns with OpenEye’s pluggable software architecture.

Hardware philosophy: Pluggability as a core OpenEye principle

The hardware architecture deliberately mirrors the OpenEye software approach:

Every hardware component is a module.

Every module can be replaced, upgraded, or extended.

The system grows horizontally, not vertically.

This ensures the OpenEye glasses remain:

Future-proof

Open-source friendly

Easy to extend by community developers

Capable of evolving alongside new edge-AI research

This design is fully supported by the direction of modern on-device AI research, which emphasizes modularity, hardware-aware intelligence, and efficient small models (TRM-style) running in constrained environments.

## Deep Dive: Omem Architecture and Implementation

The Omem (OpenEye Memory) system represents our primary contribution to edge AI memory research. This section provides a detailed technical analysis of its four-pillar architecture and the underlying implementation strategies that enable SLMs to achieve LLM-competitive performance.

### Pillar 1: Atomic Encoding — Semantic Lossless Compression

Atomic encoding transforms conversational text into self-contained, unambiguous facts. This preprocessing step is critical because SLMs lack the attention capacity to resolve ambiguities during inference—they require clean, pre-resolved input.

**Coreference Resolution**

The system maintains an entity cache of up to 50 entries, tracking:
- Entity name and type (Person, Place, Organization, Concept, Thing, Time)
- Gender inference (for pronoun resolution)
- Salience score (recently mentioned entities have higher priority)
- Temporal decay (0.95× per minute of inactivity)

Resolution follows a rule-based approach with LLM fallback:

```
Input: "He said he would fix it yesterday"
      ↓
Entity Cache: {John Smith: male, salience: 2.0}
      ↓
Rule-Based Resolution: "John Smith said John Smith would fix it yesterday"
      ↓
Bad Resolution Detection: Repeated names detected
      ↓
LLM Fallback: "John Smith said he would fix the issue yesterday"
```

The bad resolution detector uses a simple heuristic: if the same word appears 3+ times in 10 consecutive words, the rule-based output is rejected and LLM refinement is triggered.

**Temporal Anchoring**

Relative time references decay in meaning after storage. "Yesterday" becomes meaningless if retrieved weeks later. The atomic encoder converts 15+ temporal patterns to absolute dates:

| Pattern | Example | Resolution |
|---------|---------|------------|
| `today` | "I went today" | "I went on 2026-01-20" |
| `yesterday` | "He called yesterday" | "He called on 2026-01-19" |
| `last week` | "We met last week" | "We met the week of 2026-01-13" |
| `N days ago` | "3 days ago" | "2026-01-17" |
| `this morning` | "This morning" | "on 2026-01-20 morning" |

**Atomic Fact Extraction**

After coreference and temporal resolution, the LLM extracts self-contained facts using a structured format:

```
FACT|<category>|<importance>|<fact text>
```

Categories include: preference, belief, biographical, event, relationship, task, knowledge, other.

Facts with importance below the threshold (default: 0.3) are discarded. This prevents trivial statements from cluttering the memory store.

### Pillar 2: Multi-View Indexing — Beyond Semantic Similarity

Our experiments revealed that vector similarity alone achieves 68% recall. The query "What is my favorite color?" often retrieves facts about "visual preferences" or "aesthetics" rather than the exact answer. Multi-view indexing addresses this through three simultaneous retrieval paths:

**Semantic View (Weight: 0.5)**
- Dense vector embeddings using MiniLM-L6-v2 (22MB, 384 dimensions)
- DuckDB native vector operations for cosine similarity
- Captures meaning-based relationships

**Lexical View (Weight: 0.3)**
- DuckDB Full-Text Search extension (BM25 algorithm)
- Parameters: k1=1.2 (term frequency saturation), b=0.75 (length normalization)
- Recovers exact keyword matches that semantic search misses

**Symbolic View (Weight: 0.2)**
- Category-based metadata filtering
- Entity-relationship graph integration
- Enables structured queries like "facts about [person]"

The combination formula normalizes weights to sum to 1.0:
```
final_score = (semantic_weight × semantic_score) + 
              (lexical_weight × lexical_score) + 
              (symbolic_weight × symbolic_score)
```

### Pillar 3: Adaptive Retrieval Depth — Complexity-Aware Search

Fixed top-K retrieval is fundamentally mismatched to query complexity. Simple factual queries ("Where do I live?") need only K=3-5 facts, while complex reasoning queries ("How do my work and family life interact?") require K=15-20 to gather sufficient context.

**Zero-LLM Complexity Estimation**

We developed a rule-based complexity estimator that operates without LLM inference, eliminating latency overhead at query time. The estimator calculates a complexity score (0.0-1.0) using weighted factors:

| Factor | Weight | Scoring Logic |
|--------|--------|---------------|
| Entity Count | 25% | 0 entities: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4+: 0.8 |
| Question Type | 20% | Why/How: 0.8, When/Where: 0.5, What/Who: 0.3 |
| Query Length | 15% | <5 words: 0.1, 5-10: 0.3, 10-15: 0.5, 15-25: 0.7, 25+: 0.9 |
| Temporal Refs | 15% | 0: 0.0, 1: 0.3, 2: 0.5, 3+: 0.7 |
| Comparisons | 10% | "better/worse/similar": 0.4-0.8 |
| Conditionals | 10% | "if/would/could": 0.3-0.5 |
| Negations | 5% | "not/never": 0.2-0.6 |

**Dynamic Top-K Calculation**

The retrieval depth adapts using the formula:
```
k_dyn = k_base × (1 + delta × complexity)
```

With default values (k_base=5, delta=2.0):
- Simple query (complexity=0.15): K = 5 × (1 + 2×0.15) = 6.5 → 7
- Medium query (complexity=0.50): K = 5 × (1 + 2×0.50) = 10
- Complex query (complexity=0.85): K = 5 × (1 + 2×0.85) = 13.5 → 14

The result is clamped to MaxTopK (default: 20) to prevent context overflow.

**Query Classification**

The estimator also classifies queries for retrieval strategy selection:
- `factual`: What/who questions → favor exact matches
- `temporal`: When-based questions → include recency weighting
- `spatial`: Where-based questions → favor location entities
- `causal`: Why/how questions → expand context breadth
- `comparison`: Comparative queries → retrieve multiple related facts
- `open`: Open-ended queries → balanced retrieval

### Pillar 4: Rolling Summarization — Compressed Long-Term Context

The most counterintuitive finding from our research: *less is more* for long-term memory. Rather than storing all facts verbatim and retrieving them for every query, Omem maintains a compressed "user biography" that provides stable baseline context regardless of conversation length.

**Summary Update Strategy**

Two update modes are supported:

*Incremental Update* (default):
- Triggered when 5+ new facts are pending
- LLM prompt: current_summary + new_facts → updated_summary
- Preserves existing knowledge while integrating new information

*Full Regeneration*:
- Triggered manually or on configuration change
- LLM prompt: all recent facts (max 50) → fresh_summary
- Used when summary has drifted or become inconsistent

**Background Worker Architecture**

Summary refresh runs asynchronously via a background goroutine:
- Refresh interval: 5 minutes (configurable)
- Timeout per refresh: 30 seconds
- Non-blocking to main inference path
- Graceful shutdown with stop channel

**Token Budget Management**

Summaries are capped at MaxTokens (default: 512) to ensure they fit within the SLM's limited context window. The LLM prompt explicitly instructs:
- Use third person ("User prefers..." not "I prefer...")
- Focus on preferences, biography, relationships
- Omit temporary or trivial details
- Group related information together

### Pipeline Integration: Parallel Context Assembly

The pipeline orchestrates Omem alongside other context sources through parallel execution:

```
User Input
    │
    ├─→ Task A: Summarization (if history exists)
    ├─→ Task B: Vector Search (short-term memory)
    ├─→ Task C: RAG Retrieval (external knowledge)
    └─→ Task D: Omem Context (long-term personal memory)
    │
    ▼
sync.WaitGroup.Wait()
    │
    ▼
Context Builder → Merged Prompt → LLM Inference
```

This parallel execution is critical for edge devices—it hides I/O latency by running concurrent database queries and embedding operations.

**Context Assembly Order**

The final prompt is assembled in a specific order to optimize SLM attention:
1. System message (<100 tokens, behavioral constraints)
2. Omem context (user biography + retrieved facts)
3. Summary (compressed recent conversation)
4. Vector context (semantically similar past turns)
5. RAG knowledge (external documents)
6. User prompt (current query)

This ordering places the most stable, foundational context first, with query-specific context closer to the user input where SLM attention is strongest.

### Post-Inference Learning

After generating a response, the pipeline triggers asynchronous learning:

```go
// From pipeline.go
if p.omemHook != nil {
    turnID := fmt.Sprintf("%d", time.Now().UnixNano())
    p.omemHook.OnAfterGenerate(ctx, normalized, response.Text, turnID)
}
```

This non-blocking call:
1. Extracts atomic facts from the conversation turn
2. Updates the entity graph with discovered entities/relationships
3. Marks the rolling summary as dirty
4. Persists new facts to DuckDB

The async design ensures learning does not impact response latency.

