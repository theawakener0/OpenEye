# OpenEye: High-Performance Offline AI for Wearables

**OpenEye** is a transformative open-source framework designed to interact through pluggable, offline, and smart wearable systems. It empowers Small Language Models (SLMs) to achieve LLM-competitive reasoning on edge devices like Raspberry Pi 5 or Custom edge hardware, ensuring privacy, low latency, and autonomy without cloud dependence.

---

## Our Goal

The primary mission of OpenEye is to bridge the "SLM-LLM capability gap" through **architectural innovation** rather than parameter scaling. While modern AI wearables are often closed systems reliant on cloud infrastructure, OpenEye envisions:
- **Offline-First Intelligence**: Local execution preserves privacy and functions without internet.
- **Cognitive Augmentation**: Transforming smart glasses from passive displays into proactive cognitive assistants.
- **Unified Open Standard**: A modular hardware and software stack for worldwide community collaboration.

---

## Key Achievements & Findings

Our research, documented in the [Paper/](Paper/) directory, highlights several breakthroughs:

### 1. Omem (Omni Memory) Architecture
We developed **Omem**, a novel four-pillar memory architecture engineered for edge-based SLMs. It decouples knowledge from model size, allowing 1B-3B parameter models to perform complex reasoning tasks.
- **Atomic Encoding**: Uses coreference resolution and temporal anchoring to ensure stored facts remain interpretable in isolation.
- **Multi-View Indexing**: Combines semantic (vector), lexical (BM25), and symbolic (graph) retrieval, achieving **85% recall** (a 25% improvement over vector-only approaches).
- **Adaptive Retrieval**: Dynamically scales search depth based on query complexity, avoiding unnecessary LLM overhead.
- **Rolling Summarization**: Maintains long-term context while preventing linear token growth, preserving >70% long-range recall.

### 2. Efficiency Breakthroughs
- **Performance**: Omem is **4x faster** than leading alternatives like Mem0, with retrieval latencies dropping from ~50ms to **~12ms** (P50) and **~35ms** (P95) on optimized hardware.
- **Token Optimization**: Reduced storage redundancy by **~30%** through atomic deduplication and capped context growth, enabling stable, constant-time performance regardless of conversation length.
- **Edge Viability**: Verified real-time inference on **Raspberry Pi 4 (4GB) and Pi 5** using **Q4_K_M quantization** for 1B-3B parameter models.

### 3. Scientific Validations
- **Complexity-Aware Accuracy**: Validated against a 100-query corpus spanning simple lookup, multi-fact aggregation, and multi-hop reasoning. Omem maintains high recall even in "Complex" reasoning scenarios where standard RAG fails.
- **Ablation Proven**: Extensive ablation studies (8 distinct configurations) proved that the combination of **Entity Relationship Graphs** and **BM25 Lexical Search** is critical for reaching the 85% recall threshold.
- **Memory Consolidation**: Demonstrated that "Rolling Summaries" can preserve **>70% long-range recall** even after 50+ turns of conversation, effectively mimicking human-like memory consolidation.

---

## Architecture Overview

OpenEye is built with a clean separation of concerns in Go:
- **[internal/cli](internal/cli)**: Robust user-facing entry point with `chat`, `tui`, `serve`, and `memory` commands.
- **[internal/context/memory](internal/context/memory)**: Implementation of the Omem engine and adapters for various memory systems.
- **[internal/pipeline](internal/pipeline)**: The orchestration layer that handles context building and runtime execution.
- **[internal/rag](internal/rag)**: High-performance retrieval-augmented generation for local document corpora.
- **[internal/runtime](internal/runtime)**: Model-agnostic execution layer supporting various LLM backends (defaulting to llama.cpp).
- **[client/server](client/)**: TCP-based transport for connecting lightweight wearables (like ESP32) to powerful edge hosts.

---

## Getting Started

### Prerequisites
- [Go](https://go.dev/) 1.21+
- An LLM backend (e.g., [llama.cpp](https://github.com/ggerganov/llama.cpp) server running locally)

### Installation
```bash
git clone https://github.com/your-username/OpenEye.git
cd OpenEye
go build -o openeye
```

### Usage
- **Interactive Chat**: `./openeye chat --message "Who am I?"`
- **TUI Mode**: `./openeye tui`
- **CLI Mode**: `./openeye cli`
- **Background Server**: `./openeye serve` (Starts the TCP bridge for wearable clients)
- **Manage Memory**: `./openeye memory list`

---

## Roadmap

- **Phase 1 (Done)**: Functional prototype on ESP32 + Termux, Go-based pipeline, and initial Omem implementation.
- **Phase 2 (Current)**: Transition to custom PCB designs with expanded sensor arrays and enhanced power management.
- **Phase 3 (Next)**: Integration of high-performance compute modules (RPi 5 class) for fully on-device SLM inference and advanced multimodal processing.

### Future Research Directions
- **Memory Consolidation**: Implementing "offline" consolidation (resembling sleep) to improve SLM memory quality.
- **Federated Memory**: Enabling multiple edge devices to share anonymized memory patterns.
- **Cross-Session Transfer**: Improving performance by transferring episodic memories across different conversation domains.
- **Neuromorphic Integration**: Exploring spiking neural networks for optimal memory storage.


*“OpenEye aims not merely to develop another wearable device but to initiate a technological revolution... a redefinition of how humanity perceives and interacts with reality.”*
