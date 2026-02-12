# Custom Retrievers

Retrievers power OpenEye's RAG (Retrieval Augmented Generation) system. They find relevant documents or knowledge chunks from a corpus based on user queries.

## Overview

A retriever searches a knowledge corpus for information relevant to a query. OpenEye's retriever system supports:

- **Semantic Search**: Using embeddings to find similar content
- **Keyword Search**: Traditional BM25/full-text search
- **Hybrid Retrieval**: Combining semantic and keyword approaches
- **Hybrid Retrieval**: Advanced combination with re-ranking

This guide shows you how to create custom retrievers for specialized use cases.

## Interface Contract

All retrievers must implement the `rag.Retriever` interface:

```go
package rag

// Document represents a retrieved document chunk
type Document struct {
    Source  string
    Text    string
    Score   float64
    Metadata map[string]interface{}
}

// Retriever finds relevant documents from a corpus
type Retriever interface {
    // Retrieve finds documents matching the query
    Retrieve(ctx context.Context, query string, limit int) ([]Document, error)
    
    // Index adds documents to the corpus
    Index(ctx context.Context, documents []Document) error
    
    // Close releases resources
    Close() error
}
```

## Built-in Retriever Architecture

OpenEye's hybrid retriever combines multiple retrieval strategies:

```mermaid
flowchart TB
    Q["Query"]
    
    subgraph Semantic["Semantic Search"]
        VS["Vector Similarity"]
    end
    
    subgraph Keyword["Keyword Search"]
        TF["TF-IDF Scoring"]
    end
    
    subgraph Expansion["Query Expansion"]
        ET["Expanded Terms"]
    end
    
    subgraph Fusion["Score Fusion"]
        SF["Score Fusion"]
    end
    
    subgraph Ranking["Re-ranking"]
        RR["Re-ranking"]
    end
    
    subgraph Results["Final Results"]
        FR["Final Results"]
    end
    
    Q --> VS
    Q --> TF
    Q --> ET
    
    VS --> SF
    TF --> SF
    ET --> SF
    
    SF --> RR
    RR --> FR

## Custom Retriever Example: Domain-Specific Search

Here's a complete example of a custom retriever for a specific domain:

### Project Structure

```
custom-retriever/
├── go.mod
├── retriever.go
└── README.md
```

### retriever.go

```go
package customretriever

import (
    "context"
    "fmt"
    "strings"
    "sync"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/rag"
)

const defaultLimit = 10

type Retriever struct {
    corpus    map[string]*Document
    index     *InvertedIndex
    mu        sync.RWMutex
    config    Config
}

type Config struct {
    MaxResults      int
    MinScore       float64
    IndexPath      string
}

func NewRetriever(cfg Config) (*Retriever, error) {
    return &Retriever{
        corpus: make(map[string]*Document),
        index:  NewInvertedIndex(),
        config: cfg,
    }, nil
}

func (r *Retriever) Retrieve(ctx context.Context, query string, limit int) ([]rag.Document, error) {
    if limit <= 0 {
        limit = defaultLimit
    }
    if limit > r.config.MaxResults {
        limit = r.config.MaxResults
    }
    
    // Tokenize and expand query
    terms := r.tokenize(query)
    
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    // Collect candidate documents
    candidates := make(map[string]float64)
    
    for _, term := range terms {
        docs, ok := r.index[term]
        if !ok {
            continue
        }
        for docID, tfidf := range docs {
            candidates[docID] += tfidf
        }
    }
    
    // Sort and limit results
    var results []rag.Document
    for docID, score := range candidates {
        if score < r.config.MinScore {
            continue
        }
        
        doc, ok := r.corpus[docID]
        if !ok {
            continue
        }
        
        results = append(results, rag.Document{
            Source:  doc.Source,
            Text:    doc.Text,
            Score:   score,
            Metadata: doc.Metadata,
        })
    }
    
    // Sort by score descending
    sortByScore(results)
    
    if len(results) > limit {
        results = results[:limit]
    }
    
    return results, nil
}

func (r *Retriever) Index(ctx context.Context, documents []rag.Document) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    for _, doc := range documents {
        // Store document
        r.corpus[doc.Source] = &rag.Document{
            Source:  doc.Source,
            Text:    doc.Text,
            Metadata: doc.Metadata,
        }
        
        // Index terms
        terms := r.tokenize(doc.Text)
        for i, term := range terms {
            tf := float64(len(terms) - i) / float64(len(terms)) // Position-weighted TF
            r.index.Add(term, doc.Source, tf)
        }
    }
    
    return nil
}

func (r *Retriever) Close() error {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.corpus = make(map[string]*Document)
    r.index = NewInvertedIndex()
    return nil
}

func (r *Retriever) tokenize(text string) []string {
    // Simple tokenization - lowercase and split on non-alphanumeric
    text = strings.ToLower(text)
    tokens := strings.FieldsFunc(text, func(r rune) bool {
        return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
    })
    
    // Remove stopwords (simplified)
    stopwords := map[string]bool{
        "the": true, "a": true, "an": true, "and": true, "or": true,
        "but": true, "in": true, "on": true, "at": true, "to": true,
        "for": true, "of": true, "with": true, "by": true,
    }
    
    var filtered []string
    for _, token := range tokens {
        if !stopwords[token] && len(token) > 1 {
            filtered = append(filtered, token)
        }
    }
    
    return filtered
}

func sortByScore(docs []rag.Document) {
    sort.Slice(docs, func(i, j int) bool {
        return docs[i].Score > docs[j].Score
    })
}

// InvertedIndex for keyword search
type InvertedIndex map[string]map[string]float64

func NewInvertedIndex() InvertedIndex {
    return make(InvertedIndex)
}

func (i InvertedIndex) Add(term, docID string, weight float64) {
    if i[term] == nil {
        i[term] = make(map[string]float64)
    }
    i[term][docID] += weight
}
```

## Hybrid Retrieval Implementation

Here's how to combine semantic and keyword search:

```go
package hybridretriever

import (
    "context"
    "math"
    "strings"
    "sync"

    "OpenEye/internal/embedding"
    "OpenEye/internal/rag"
)

type HybridRetriever struct {
    keywordRetriever KeywordRetriever
    semanticRetriever SemanticRetriever
    embedder        embedding.Provider
    semanticWeight  float64
    keywordWeight   float64
    config          Config
}

func (r *HybridRetriever) Retrieve(ctx context.Context, query string, limit int) ([]rag.Document, error) {
    // Get keyword results
    keywordResults, err := r.keywordRetriever.Retrieve(ctx, query, limit*2)
    if err != nil {
        return nil, err
    }
    
    // Get semantic results
    semanticResults, err := r.semanticRetriever.Retrieve(ctx, query, limit*2)
    if err != nil {
        return nil, err
    }
    
    // Normalize scores
    keywordScores := normalizeScores(keywordResults)
    semanticScores := normalizeScores(semanticResults)
    
    // Merge scores
    merged := make(map[string]float64)
    
    for doc, score := range keywordScores {
        merged[doc.Source] = r.keywordWeight * score
    }
    
    for doc, score := range semanticScores {
        merged[doc.Source] += r.semanticWeight * score
    }
    
    // Reconstruct documents
    docMap := make(map[string]rag.Document)
    for _, doc := range keywordResults {
        docMap[doc.Source] = doc
    }
    for _, doc := range semanticResults {
        if _, ok := docMap[doc.Source]; !ok {
            docMap[doc.Source] = doc
        }
    }
    
    // Sort and limit
    var results []rag.Document
    for source, score := range merged {
        doc := docMap[source]
        doc.Score = score
        results = append(results, doc)
    }
    
    sortByScore(results)
    
    if len(results) > limit {
        results = results[:limit]
    }
    
    return results, nil
}

func normalizeScores(docs []rag.Document) map[string]float64 {
    maxScore := 0.0
    for _, doc := range docs {
        if doc.Score > maxScore {
            maxScore = doc.Score
        }
    }
    
    scores := make(map[string]float64)
    if maxScore == 0 {
        return scores
    }
    
    for _, doc := range docs {
        scores[doc.Source] = doc.Score / maxScore
    }
    
    return scores
}
```

## Configuration Integration

### RAG Configuration Structure

```go
type RAGConfig struct {
    Enabled        bool     `yaml:"enabled"`
    CorpusPath     string   `yaml:"corpus_path"`
    MaxChunks      int      `yaml:"max_chunks"`
    ChunkSize      int      `yaml:"chunk_size"`
    ChunkOverlap   int      `yaml:"chunk_overlap"`
    MinScore       float64  `yaml:"min_score"`
    IndexPath      string   `yaml:"index_path"`
    Extensions     []string `yaml:"extensions"`
    
    // Hybrid retrieval settings
    HybridEnabled        bool    `yaml:"hybrid_enabled"`
    MaxCandidates        int     `yaml:"max_candidates"`
    DiversityThreshold   float64 `yaml:"diversity_threshold"`
    SemanticWeight       float64 `yaml:"semantic_weight"`
    KeywordWeight        float64 `yaml:"keyword_weight"`
    EnableQueryExpansion bool    `yaml:"enable_query_expansion"`
}
```

### YAML Configuration

```yaml
rag:
  enabled: true
  corpus_path: "./knowledge_base"
  max_chunks: 4
  chunk_size: 512
  chunk_overlap: 64
  min_score: 0.2
  index_path: "./rag_index"
  extensions: [".txt", ".md", ".pdf"]
  
  # Hybrid retrieval
  hybrid_enabled: true
  semantic_weight: 0.7
  keyword_weight: 0.3
  enable_query_expansion: true
```

## Document Chunking Strategies

### Fixed-Size Chunking

```go
type FixedChunkStrategy struct {
    chunkSize   int
    chunkOverlap int
}

func (s *FixedChunkStrategy) Chunk(documents []Document) []Document {
    var chunks []Document
    
    for _, doc := range documents {
        text := doc.Text
        offset := 0
        
        for offset < len(text) {
            end := offset + s.chunkSize
            if end > len(text) {
                end = len(text)
            }
            
            chunk := Document{
                Source:  doc.Source,
                Text:    text[offset:end],
                Metadata: copyMetadata(doc.Metadata),
            }
            chunks = append(chunks, chunk)
            
            offset = end - s.chunkOverlap
            if offset >= len(text) {
                break
            }
        }
    }
    
    return chunks
}
```

### Semantic Chunking

```go
type SemanticChunkStrategy struct {
    embedder embedding.Provider
    minSimilarity float64
    maxChunkSize  int
}

func (s *SemanticChunkStrategy) Chunk(documents []Document) ([]Document, error) {
    var chunks []Document
    
    for _, doc := range documents {
        sentences := splitIntoSentences(doc.Text)
        
        var currentChunk []string
        var currentVector []float32
        
        for i, sentence := range sentences {
            sentenceVector, err := s.embedder.Embed(context.Background(), sentence)
            if err != nil {
                return nil, err
            }
            
            // Check similarity with current chunk
            if len(currentChunk) > 0 && i < len(sentences)-1 {
                similarity := cosineSimilarity(currentVector, sentenceVector)
                if similarity < s.minSimilarity {
                    // Start new chunk
                    chunks = append(chunks, createChunk(doc, currentChunk))
                    currentChunk = nil
                    currentVector = nil
                }
            }
            
            currentChunk = append(currentChunk, sentence)
            
            if len(currentChunk) == 1 {
                currentVector = sentenceVector
            } else {
                currentVector = averageVectors(currentVector, sentenceVector)
            }
            
            // Check size limit
            if len(strings.Join(currentChunk, " ")) > s.maxChunkSize {
                chunks = append(chunks, createChunk(doc, currentChunk))
                currentChunk = nil
                currentVector = nil
            }
        }
        
        // Add remaining chunk
        if len(currentChunk) > 0 {
            chunks = append(chunks, createChunk(doc, currentChunk))
        }
    }
    
    return chunks, nil
}
```

## Index Management

### Persistent Index

```go
type PersistentRetriever struct {
    index   *HybridIndex
    config  Config
}

func (r *PersistentRetriever) SaveIndex(path string) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    file, err := os.Create(path)
    if err != nil {
        return fmt.Errorf("failed to create index file: %w", err)
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    if err := encoder.Encode(r.index); err != nil {
        return fmt.Errorf("failed to encode index: %w", err)
    }
    
    return nil
}

func (r *PersistentRetriever) LoadIndex(path string) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    file, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("failed to open index file: %w", err)
    }
    defer file.Close()
    
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&r.index); err != nil {
        return fmt.Errorf("failed to decode index: %w", err)
    }
    
    return nil
}
```

## Security Considerations

### 1. Path Traversal Prevention

```go
func (r *Retriever) loadDocument(path string) (string, error) {
    // Resolve to absolute path
    absPath, err := filepath.Abs(path)
    if err != nil {
        return "", fmt.Errorf("invalid path: %w", err)
    }
    
    // Ensure path is within allowed directory
    allowedDir := r.config.CorpusPath
    if !strings.HasPrefix(absPath, allowedDir) {
        return "", errors.New("path outside allowed corpus directory")
    }
    
    // Read file
    content, err := os.ReadFile(absPath)
    if err != nil {
        return "", fmt.Errorf("failed to read file: %w", err)
    }
    
    return string(content), nil
}
```

### 2. Document Validation

```go
func (r *Retriever) ValidateDocument(doc rag.Document) error {
    if doc.Source == "" {
        return errors.New("document source is required")
    }
    
    if len(doc.Text) > maxDocumentSize {
        return fmt.Errorf("document exceeds maximum size of %d bytes", maxDocumentSize)
    }
    
    // Check for suspicious patterns
    if containsNullBytes(doc.Text) {
        return errors.New("document contains null bytes")
    }
    
    return nil
}
```

### 3. Query Validation

```go
func (r *Retriever) ValidateQuery(query string) error {
    if strings.TrimSpace(query) == "" {
        return errors.New("query cannot be empty")
    }
    
    if len(query) > maxQueryLength {
        return fmt.Errorf("query exceeds maximum length of %d characters", maxQueryLength)
    }
    
    // Check for injection patterns
    if containsSQLInjection(query) {
        return errors.New("query contains suspicious patterns")
    }
    
    return nil
}
```

## Performance Optimization

### 1. Caching

```go
type CachedRetriever struct {
    retriever rag.Retriever
    cache     *lru.Cache[string, []rag.Document]
    mu        sync.RWMutex
}

func (c *CachedRetriever) Retrieve(ctx context.Context, query string, limit int) ([]rag.Document, error) {
    cacheKey := generateCacheKey(query, limit)
    
    // Check cache
    c.mu.RLock()
    if cached, ok := c.cache.Get(cacheKey); ok {
        c.mu.RUnlock()
        return cached, nil
    }
    c.mu.RUnlock()
    
    // Get from underlying retriever
    results, err := c.retriever.Retrieve(ctx, query, limit)
    if err != nil {
        return nil, err
    }
    
    // Cache results
    c.mu.Lock()
    c.cache.Add(cacheKey, results)
    c.mu.Unlock()
    
    return results, nil
}
```

### 2. Parallel Retrieval

```go
func (r *HybridRetriever) RetrieveParallel(ctx context.Context, query string, limit int) ([]rag.Document, error) {
    var wg sync.WaitGroup
    var keywordResults []rag.Document
    var semanticResults []rag.Document
    var keywordErr, semanticErr error
    
    // Keyword search in parallel
    wg.Add(1)
    go func() {
        defer wg.Done()
        keywordResults, keywordErr = r.keywordRetriever.Retrieve(ctx, query, limit)
    }()
    
    // Semantic search in parallel
    wg.Add(1)
    go func() {
        defer wg.Done()
        semanticResults, semanticErr = r.semanticRetriever.Retrieve(ctx, query, limit)
    }()
    
    wg.Wait()
    
    if keywordErr != nil {
        return nil, keywordErr
    }
    if semanticErr != nil {
        return nil, semanticErr
    }
    
    return r.fuseResults(keywordResults, semanticResults, limit), nil
}
```

## Testing Retrievers

### Unit Testing

```go
func TestRetriever_Retrieve(t *testing.T) {
    retriever, err := NewRetriever(Config{
        MaxResults: 10,
        MinScore:   0.1,
    })
    if err != nil {
        t.Fatalf("Failed to create retriever: %v", err)
    }
    defer retriever.Close()
    
    // Index documents
    docs := []rag.Document{
        {Source: "doc1", Text: "OpenEye is a framework for building AI applications"},
        {Source: "doc2", Text: "Plugins extend OpenEye's functionality"},
    }
    if err := retriever.Index(context.Background(), docs); err != nil {
        t.Fatalf("Index failed: %v", err)
    }
    
    // Retrieve
    results, err := retriever.Retrieve(context.Background(), "What is OpenEye?", 5)
    if err != nil {
        t.Fatalf("Retrieve failed: %v", err)
    }
    
    if len(results) == 0 {
        t.Error("Expected results, got none")
    }
    
    if len(results) > 0 && results[0].Source != "doc1" {
        t.Errorf("Expected doc1 as top result, got %s", results[0].Source)
    }
}
```

### Integration Testing

```go
func TestRetriever_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    retriever, err := NewRetriever(Config{
        CorpusPath: "./test_corpus",
        MaxResults: 20,
    })
    if err != nil {
        t.Fatalf("Failed to create retriever: %v", err)
    }
    defer retriever.Close()
    
    // Index corpus
    documents := loadTestCorpus("./test_corpus")
    if err := retriever.Index(context.Background(), documents); err != nil {
        t.Fatalf("Index failed: %v", err)
    }
    
    // Run queries
    queries := []string{
        "What is the main topic?",
        "How does it work?",
    }
    
    for _, query := range queries {
        results, err := retriever.Retrieve(context.Background(), query, 5)
        if err != nil {
            t.Errorf("Query '%s' failed: %v", query, err)
        }
        
        if len(results) == 0 {
            t.Errorf("No results for query: %s", query)
        }
    }
}
```

## Common Issues

### Empty Results

```go
func (r *Retriever) Retrieve(ctx context.Context, query string, limit int) ([]rag.Document, error) {
    results, err := r.doRetrieve(ctx, query, limit)
    if err != nil {
        return nil, err
    }
    
    // Handle empty results
    if len(results) == 0 {
        // Log warning
        log.Printf("[retriever] No results for query: %s", query)
        
        // Return empty slice (not nil)
        return []rag.Document{}, nil
    }
    
    return results, nil
}
```

### Index Not Updated

```go
func (r *Retriever) Index(ctx context.Context, documents []rag.Document) error {
    r.mu.Lock()
    defer r.mu.Unlock()
    
    // Verify all documents
    for _, doc := range documents {
        if err := r.ValidateDocument(doc); err != nil {
            return fmt.Errorf("invalid document %s: %w", doc.Source, err)
        }
    }
    
    // Perform indexing
    for _, doc := range documents {
        r.indexDocument(doc)
    }
    
    // Save index
    if err := r.SaveIndex(r.config.IndexPath); err != nil {
        return fmt.Errorf("failed to save index: %w", err)
    }
    
    return nil
}
```

## Checklist for Production

- [ ] Interface implemented correctly
- [ ] Document chunking strategy defined
- [ ] Index persistence implemented
- [ ] Query validation implemented
- [ ] Path traversal prevention
- [ ] Document validation
- [ ] Performance caching
- [ ] Error handling complete
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Benchmarks run

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Embedding Providers](embedding-providers.md)
- [Memory Engines](memory-engines.md)
- [Best Practices](best-practices.md)
