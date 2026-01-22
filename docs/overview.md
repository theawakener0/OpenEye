# OpenEye Framework Overview

OpenEye is a lightweight on-device small language model (SLM) framework. It focuses on fast feedback loops, deterministic configuration, and a clean separation between transport, context building, memory, and runtime inference.

## Core Components

### CLI (`internal/cli`)
The CLI is the user-facing entry point. It wires subcommands (`chat`, `serve`, `memory`) to the shared pipeline and loads configuration before delegating work.

### Pipeline (`internal/pipeline`)
The pipeline assembles context, retrieves conversation history, invokes the runtime adapter, manages multi-tier memory (including Omem), handles vision tasks, and persists new turns. It hides implementation details from both interactive and long-running workflows.

### Retrieval (`internal/rag`)
The retriever indexes the configured knowledge corpus into semantic chunks, supports configurable chunking overlap, and skips files outside the allowed extension list. It persists cosine-normalized vectors so subsequent runs can reuse a cached index.

### Embeddings (`internal/embedding`)
Embedding providers wrap external services (llama.cpp by default) and expose a narrow `Provider` interface used by RAG, the summarizer, and advanced memory engines (Omem/Mem0). Timeouts and model selection are configurable per backend.

### Vision (`internal/image`)
The image processor provides multimodal capabilities, allowing the pipeline to ingest, resize, and normalize images before passing them to vision-capable SLMs.

### Summarizer Assistant (`internal/pipeline/summarizer.go`)
An optional summariser selects the most relevant turns via embeddings, enforces similarity thresholds, and caches vectors so repeated summaries avoid redundant requests.

### Runtime Manager (`internal/runtime`)
The runtime package normalizes inference requests. Backends register themselves in a registry and implement the `Adapter` interface. The default HTTP adapter lives in `internal/llmclient`.

### Multi-tier Memory (`internal/context/memory`)
OpenEye implements a sophisticated multi-tier memory architecture to provide long-term coherence on resource-constrained devices:

1. **Short-term Memory**: A SQLite-backed store for recent dialog turns, using a sliding window context.
2. **Vector Memory**: A DuckDB-backed vector store that provides semantically relevant turn retrieval based on embedding similarity.
3. **Mem0**: A fact-based long-term memory system that extracts and updates individual facts from the conversation.
4. **Omem (OpenEye Memory)**: A next-generation memory engine combining atomic fact encoding, multi-view indexing (semantic, lexical, and symbolic), and a lightweight entity-relationship graph for high-precision retrieval and temporal reasoning.

### Transport (`server` and `client`)
- `server/tcp_server.go` exposes a TCP listener that accepts newline-delimited messages, acknowledges receipt, and responds with base64 encoded `RESP` or `ERR` frames once the pipeline returns.
- `client/tcp_client.go` wraps connection management and understands the same protocol. `SendAndReceive` is the high-level helper that takes care of the entire round trip.

## Request Lifecycle

1. **Input** – The CLI takes a prompt (`chat`) or the TCP server accepts a message from a client. Images are processed and normalized if present.
2. **Context Build** – The pipeline normalizes the message, gathers recent memory entries, retrieves facts from Omem or Vector Memory, loads the system prompt, and renders the conversation template.
3. **Runtime Selection** – The runtime manager instantiates the configured adapter (HTTP by default) and forwards the formatted prompt along with generation hints.
4. **Model Invocation** – The adapter issues an HTTP request to the local LLM endpoint (or other adapters in the future) and returns a structured response.
5. **Streaming (Optional)** – When streaming is requested, callbacks receive incremental tokens or chunks. The pipeline buffers them before persisting.
6. **Persistence** – User and assistant turns are appended to the short-term memory store. New facts are extracted and indexed by Omem for long-term recall.
7. **Transport Response** – CLI prints the final text; the TCP server packages the reply and ships it back to the originating client.

## Configuration

Configuration is read in the following order:
1. Built-in defaults (`internal/config/config.go`).
2. Optional YAML file (`openeye.yaml` by default or path in `APP_CONFIG`).
3. Environment variables (take precedence over file values).

### YAML Structure

```yaml
runtime:
  backend: http
  http:
    base_url: "http://127.0.0.1:42069"
    timeout: "60s"
  defaults:
    max_tokens: 512
    temperature: 0.7
    top_k: 40
    top_p: 0.9
    min_p: 0.05
    repeat_penalty: 1.1
    repeat_last_n: 64
memory:
  path: "openeye_memory.db"
  turns_to_use: 6
server:
  host: "127.0.0.1"
  port: 42067
  enabled: true
conversation:
  system_message: ""  # optional override
  template_path: ""   # optional path to custom Markdown template

rag:
  enabled: false
  corpus_path: "./rag_corpus"
  max_chunks: 4
  chunk_size: 512
  chunk_overlap: 64
  min_score: 0.2
  index_path: "openeye_rag.index"
  extensions: [".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".pdf"]

image:
  enabled: false
  max_width: 800
  max_height: 600
  quality: 85
  output_format: "jpeg"

assistants:
  summarizer:
    enabled: false
    prompt: "Summarize the following conversation history in concise bullet points highlighting commitments, open questions, and facts. Respond with at most five bullets."
    max_tokens: 128
    min_turns: 3
    max_references: 8
    similarity_threshold: 0.1
    max_transcript_tokens: 0

embedding:
  enabled: false
  backend: "llamacpp"
  llamacpp:
    base_url: "http://127.0.0.1:8080"
    model: ""
    timeout: "30s"
```

### Environment Overrides

**Core runtime**

| Variable             | Description                              |
|----------------------|------------------------------------------|
| `APP_CONFIG`         | Path to a YAML file to load.             |
| `APP_LLM_BASEURL`    | Overrides `runtime.http.base_url`.       |
| `APP_MEMORY_PATH`    | Overrides `memory.path`.                 |
| `APP_MEMORY_TURNS`   | Overrides `memory.turns_to_use`.         |
| `APP_SYSMSG`         | Overrides `conversation.system_message`. |
| `APP_CONTEXT_PATH`   | Overrides `conversation.template_path`.  |
| `APP_SERVER_HOST`    | Overrides `server.host`.                 |
| `APP_SERVER_PORT`    | Overrides `server.port`.                 |
| `APP_SERVER_ENABLED` | Enables or disables the TCP server.      |

**Retrieval helpers**

| Variable               | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| `APP_RAG_ENABLED`      | Toggles retrieval augmented generation globally.                   |
| `APP_RAG_CORPUS`       | Filesystem path to the corpus directory.                            |
| `APP_RAG_MAXCHUNKS`    | Overrides `rag.max_chunks` (number of results to surface).         |
| `APP_RAG_CHUNKSIZE`    | Overrides `rag.chunk_size` (words per chunk).                       |
| `APP_RAG_CHUNKOVERLAP` | Overrides `rag.chunk_overlap` (word overlap between chunks).        |
| `APP_RAG_MINSCORE`     | Sets the cosine similarity threshold for matches.                  |
| `APP_RAG_INDEX`        | Path to the persisted vector index file.                            |
| `APP_RAG_EXTENSIONS`   | Comma-separated list of allowed file extensions (e.g. `.md,.pdf`). |

**Summarizer assistant**

| Variable                           | Description                                                   |
|------------------------------------|---------------------------------------------------------------|
| `APP_SUMMARY_ENABLED`              | Enables or disables the summarizer helper.                    |
| `APP_SUMMARY_PROMPT`               | Custom prompt template for summary generation.                |
| `APP_SUMMARY_MAXTOKENS`            | Overrides `assistants.summarizer.max_tokens`.                 |
| `APP_SUMMARY_MIN_TURNS`            | Minimum conversation turns required before summarising.       |
| `APP_SUMMARY_MAX_REFERENCES`       | Maximum history snippets to feed into the summary prompt.     |
| `APP_SUMMARY_SIMILARITY`           | Similarity threshold for including a turn in the summary.     |
| `APP_SUMMARY_MAX_TRANSCRIPT_TOKENS`| Cap on transcript tokens considered when summarising.        |

**Embedding provider**

| Variable                  | Description                                             |
|---------------------------|---------------------------------------------------------|
| `APP_EMBEDDING_ENABLED`   | Enables embedding calls for RAG and summarization.     |
| `APP_EMBEDDING_BACKEND`   | Backend identifier (`llamacpp` by default).            |
| `APP_EMBEDDING_BASEURL`   | Base URL for the embedding service.                    |
| `APP_EMBEDDING_MODEL`     | Embedding model name to request (optional).            |
| `APP_EMBEDDING_TIMEOUT`   | Request timeout (duration string, e.g. `45s`).         |

## Retrieval Augmented Generation

Enable RAG by setting `rag.enabled: true` (or `APP_RAG_ENABLED=true`) and pointing `rag.corpus_path` to a directory of source documents. OpenEye currently ingests plain text, Markdown, CSV/TSV, JSON/YAML, logs, and PDFs; adjust the whitelist with `rag.extensions` or `APP_RAG_EXTENSIONS` when you need extra formats. The retriever chunkifies documents using the configured `rag.chunk_size` and `rag.chunk_overlap`, embeds each slice, and stores normalized vectors in memory.

To avoid re-indexing large corpora on every run, set `rag.index_path` to a writable file. The retriever writes a checksum-tagged cache that is transparently invalidated when files change. You can temporarily bypass retrieval per request using the CLI flag `--no-rag`, or reduce/expand the number of returned chunks with `--rag-limit`.

Example indexing and query flow:

```bash
# Populate the corpus with PDFs, Markdown, CSVs, etc.
cp ./docs/*.pdf ./rag_corpus/

# Enable retrieval and request with a stricter limit
OpenEye chat --message "Summarise the support policy" --rag-limit 6
```

## Conversation Summaries

The summarizer assistant provides structured rollups of recent turns to keep longer sessions manageable. Enable it with `assistants.summarizer.enabled: true` and ensure embeddings are available. The helper selects up to `max_references` turns whose similarity clears `similarity_threshold`, optionally truncating the transcript via `max_transcript_tokens`. Minimum turn guards (`min_turns`) prevent premature summaries.

At runtime you can opt out with `--no-summary`. Custom prompts or token budgets are best supplied through the config file or the `APP_SUMMARY_*` environment variables listed above.

## Advanced Long-term Memory (Omem)

OpenEye's premier memory system, Omem, enables deep context and long-term personalization. Unlike simple sliding windows, Omem:
- **Extracts Atomic Facts**: Deconstructs conversation turns into high-fidelity "atomic facts".
- **Multi-view Indexing**: Indexes facts using semantic (vector), lexical (full-text), and symbolic (keywords) views for reliable retrieval regardless of query style.
- **Lightweight Entity Graph**: Maintains relationships between entities (people, places, concepts) to support complex reasoning over the conversational history.
- **Adaptive Retrieval**: Adjusts retrieval strategies based on query complexity to balance speed and accuracy.

To enable Omem, set `memory.omem.enabled: true` in your configuration.

## Multimodal Vision Support

OpenEye supports vision tasks by integrating an image processor in the pipeline. When an image is provided as part of a prompt, the framework:
1. Validates and decodes the image (JPEG, PNG, BMP).
2. Resizes and normalizes it according to `image` configuration settings.
3. Encodes it for the vision-capable backend (e.g., LLaVA or other multimodal adapters).

Enable vision features by setting `image.enabled: true`.

## Semantic Embeddings

Both RAG, the summarizer, and advanced memory engines rely on the shared embedding provider configured under `embedding`. The default `llamacpp` backend calls a llama.cpp HTTP server; override `embedding.llamacpp.base_url`, `model`, and `timeout` to match your deployment. Flip `embedding.enabled: true` (or export `APP_EMBEDDING_ENABLED=true`) so the pipeline can request vectors. If the embedding service is unreachable, retrieval and summarisation are automatically skipped while logging the error.

## Benchmarking & Performance

OpenEye includes a dedicated benchmarking suite (`internal/context/memory/benchmark`) to evaluate and optimize memory engines. It measures:
- **Latency**: Write and retrieval speeds across different engine types.
- **Recall Accuracy**: Ability to retrieve "planted facts" from large conversational histories.
- **Token Efficiency**: Optimization of context window usage.
- **Resource Footprint**: Memory and CPU usage on resource-constrained devices.

Run benchmarks using the internal testing tools to compare Omem, Mem0, and Classic memory performance on your specific hardware.

## Running the CLI

The `chat` command accepts per-request controls: `--stream` for incremental tokens, `--no-rag` and `--no-summary` to bypass helpers, and `--rag-limit` to override the number of retrieved chunks.

### One-off Prompt
```bash
OpenEye chat --message "Write a haiku about local inference."
```

### Streaming Response
```bash
OpenEye chat --message "Explain streaming tokens." --stream
```

### Start TCP Server
```bash
OpenEye serve
```
The server logs incoming prompts and pushes responses back to clients over the RESP/ERR protocol.

### Inspect Memory
```bash
OpenEye memory --n 12
```

### Command Examples

Prepare a bespoke configuration, launch the TCP server, and send a round-trip request from the bundled client:

```bash
# 1. Export a custom configuration file path
export APP_CONFIG=$PWD/openeye.yaml

# 2. Start the TCP server in one terminal
OpenEye serve

# 3. In another terminal, run the Go sample client
go run ./cmd/examples/tcpclient
```

If you prefer a one-shot interaction without starting the server, the CLI can be chained with shell tools:

```bash
echo "Plan a 3-step refactor checklist" | xargs -0 OpenEye chat --stream --message
```

## Writing Clients

Minimal client using the provided package:
```go
package main

import (
    "log"

    "OpenEye/client"
)

func main() {
    tcp := client.NewTCPClient("127.0.0.1", "42067")
    if err := tcp.Connect(); err != nil {
        log.Fatal(err)
    }
    defer tcp.Disconnect()

    resp, err := tcp.SendAndReceive("## Prompt\n\nTell me a story.")
    if err != nil {
        log.Fatal(err)
    }

    log.Println("assistant:", resp)
}
```

### Framework Usage Example

When embedding OpenEye in a larger application, wire the pipeline directly and reuse its orchestration without the CLI layer:

```go
package main

import (
  "context"
  "log"

  "OpenEye/internal/config"
  "OpenEye/internal/pipeline"
  "OpenEye/internal/runtime"
)

func main() {
  cfg, err := config.Resolve()
  if err != nil {
    log.Fatal(err)
  }

  pipe, err := pipeline.New(cfg, runtime.DefaultRegistry)
  if err != nil {
    log.Fatal(err)
  }
  defer pipe.Close()

  result, err := pipe.Respond(context.Background(), "Summarize the latest design notes", pipeline.Options{})
  if err != nil {
    log.Fatal(err)
  }

  log.Println("assistant:", result.Text)
}
```

The snippet loads configuration the same way the CLI does, then invokes the pipeline once to obtain an assistant reply. Swap `pipeline.Options{Stream: true, ...}` to integrate streaming callbacks or attach custom generation hints as needed.

## Extending OpenEye

- **Add a backend** – Implement `runtime.Adapter`, register it in `internal/runtime/registry.go`, and supply configuration fields for it.
- **Custom templates** – Point `conversation.template_path` to a Markdown file and structure it as needed.
- **Alternative storage** – Replace `internal/context/memory` with another implementation and update the pipeline wiring accordingly.

## Directory Reference

```
client/        TCP client library
internal/
  cli/         CLI entry points
  config/      Unified configuration loader
  context/     Context and formatting logic
    memory/    Multi-tier memory systems (Short-term, Vector, Omem, Mem0)
    omem/      OpenEye Memory next-gen implementation
    mem0/      Legacy fact-based memory adapter
  embedding/   Embedding providers (llama.cpp HTTP adapter by default)
  image/       Image processing for multimodal support
  llmclient/   HTTP adapter for LLM servers
  pipeline/    High-level orchestration
  rag/         Retrieval helpers and vector index management
  runtime/     Backend registry and interfaces
server/        TCP server implementation
rag_corpus/    Default on-disk knowledge base scanned by the retriever
```

## Troubleshooting

- Verify the underlying LLM server (default `http://127.0.0.1:42069`) is running; otherwise the adapter logs an error while preserving the conversation state.
- Increase `memory.turns_to_use` when building longer contexts on devices with more memory.
- Enable debug logging by setting Go's `GODEBUG` as needed when inspecting network behavior.
