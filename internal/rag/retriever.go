package rag

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"

	"container/heap"
	pdf "github.com/ledongthuc/pdf"
)

type scoredResult struct {
	chunk vectorChunk
	score float64
	index int
}

type minHeap []scoredResult

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *minHeap) Push(x any) {
	*h = append(*h, x.(scoredResult))
}

func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// Document represents a single piece of retrieved knowledge.
type Document struct {
	ID     string
	Source string
	Text   string
	Score  float64
}

// Retriever exposes RAG retrieval capabilities.
type Retriever interface {
	Retrieve(ctx context.Context, query string, limit int) ([]Document, error)
	Close() error
}

type vectorChunk struct {
	id     string
	source string
	text   string
	vector []float32
}

var defaultRAGExtensions = []string{
	".txt",
	".md",
	".markdown",
	".rst",
	".log",
	".csv",
	".tsv",
	".json",
	".yaml",
	".yml",
	".pdf",
}

// FilesystemRetriever performs document retrieval over local files using semantic embeddings.
type FilesystemRetriever struct {
	chunks              []vectorChunk
	minScore            float64
	earlyTermination    bool
	earlyTermMultiplier float64
	earlyTermMinChunks  int
	mu                  sync.RWMutex
	embedder            embedding.Provider
}

// NewFilesystemRetriever builds a retriever that indexes plain-text files on disk into a lightweight vector store.
func NewFilesystemRetriever(cfg config.RAGConfig, embedder embedding.Provider) (*FilesystemRetriever, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	if embedder == nil {
		return nil, errors.New("rag: embedding provider required when RAG is enabled")
	}
	if strings.TrimSpace(cfg.CorpusPath) == "" {
		return nil, errors.New("rag: corpus_path is required when RAG is enabled")
	}

	info, err := os.Stat(cfg.CorpusPath)
	if err != nil {
		return nil, fmt.Errorf("rag: failed to stat corpus path: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("rag: corpus_path %q is not a directory", cfg.CorpusPath)
	}

	chunkSize := cfg.ChunkSize
	if chunkSize <= 0 {
		chunkSize = 512
	}
	overlap := cfg.ChunkOverlap
	if overlap < 0 {
		overlap = 0
	}

	retriever := &FilesystemRetriever{
		minScore:            cfg.MinScore,
		embedder:            embedder,
		earlyTermination:    cfg.EarlyTermination,
		earlyTermMultiplier: cfg.EarlyTermMultiplier,
		earlyTermMinChunks:  cfg.EarlyTermMinChunks,
	}
	if retriever.minScore <= 0 {
		retriever.minScore = 0.2
	}
	if retriever.earlyTermMultiplier <= 0 {
		retriever.earlyTermMultiplier = 3.0
	}
	if retriever.earlyTermMinChunks <= 0 {
		retriever.earlyTermMinChunks = 500
	}

	allowedExt := normalizeExtensions(cfg.Extensions)
	if len(allowedExt) == 0 {
		allowedExt = normalizeExtensions(defaultRAGExtensions)
	}

	indexPath := strings.TrimSpace(cfg.IndexPath)
	checksum, err := checksumCorpus(cfg.CorpusPath)
	if err != nil {
		return nil, err
	}

	cached, err := loadIndex(indexPath, checksum)
	if err == nil && cached != nil {
		retriever.chunks = cached
		return retriever, nil
	}

	chunks, err := buildIndex(cfg.CorpusPath, chunkSize, overlap, embedder, allowedExt)
	if err != nil {
		return nil, err
	}
	retriever.chunks = chunks

	if indexPath != "" {
		if err := persistIndex(indexPath, checksum, chunks); err != nil {
			log.Printf("warning: rag failed to persist index %q: %v", indexPath, err)
		}
	}

	return retriever, nil
}

// Retrieve returns the most relevant chunks for the provided query.
func (r *FilesystemRetriever) Retrieve(ctx context.Context, query string, limit int) ([]Document, error) {
	if r == nil {
		return nil, nil
	}
	normalized := strings.TrimSpace(query)
	if normalized == "" {
		return nil, nil
	}

	if limit <= 0 {
		limit = 4
	}

	r.mu.RLock()
	chunks := make([]vectorChunk, len(r.chunks))
	copy(chunks, r.chunks)
	r.mu.RUnlock()

	if len(chunks) == 0 {
		return nil, nil
	}

	queryVec, err := r.embedder.Embed(ctx, normalized)
	if err != nil {
		return nil, err
	}
	if len(queryVec) != len(chunks[0].vector) {
		return nil, fmt.Errorf("rag: embedding dimension mismatch (got %d, expected %d)", len(queryVec), len(chunks[0].vector))
	}
	normalizeVector(queryVec)

	useEarlyTermination := r.earlyTermination && len(chunks) >= r.earlyTermMinChunks
	targetResults := limit
	if useEarlyTermination {
		targetResults = int(float64(limit) * r.earlyTermMultiplier)
		if targetResults < limit*2 {
			targetResults = limit * 2
		}
	}

	h := &minHeap{}
	heap.Init(h)
	worstScore := 0.0
	processed := 0
	batchSize := 100

	for i, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(chunk.vector) == 0 {
			continue
		}
		processed++

		score := cosineSimilarity(queryVec, chunk.vector)
		if score < r.minScore {
			continue
		}

		if useEarlyTermination && h.Len() >= targetResults && score <= worstScore {
			continue
		}

		heap.Push(h, scoredResult{chunk: chunk, score: score, index: i})

		if useEarlyTermination && h.Len() > targetResults {
			heap.Pop(h)
		}

		if h.Len() > 0 {
			worstScore = (*h)[0].score
		}

		if useEarlyTermination && processed >= r.earlyTermMinChunks && h.Len() >= targetResults {
			remaining := len(chunks) - processed
			maxPossibleScore := 1.0
			potentialBest := worstScore + float64(remaining)*maxPossibleScore*0.001
			if processed%batchSize == 0 && potentialBest <= worstScore*1.1 {
				log.Printf("rag: early termination at chunk %d/%d (results: %d, worst: %.3f)", processed, len(chunks), h.Len(), worstScore)
				break
			}
		}
	}

	results := make([]scoredResult, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(scoredResult)
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].score == results[j].score {
			return results[i].chunk.id < results[j].chunk.id
		}
		return results[i].score > results[j].score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	docs := make([]Document, 0, len(results))
	for _, res := range results {
		docs = append(docs, Document{
			ID:     res.chunk.id,
			Source: res.chunk.source,
			Text:   res.chunk.text,
			Score:  res.score,
		})
	}

	return docs, nil
}

// Close currently releases no additional resources.
func (r *FilesystemRetriever) Close() error {
	return nil
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func normalizeVector(v []float32) {
	var norm float64
	for _, val := range v {
		norm += float64(val * val)
	}
	if norm == 0 {
		return
	}
	inv := 1 / float32(math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
}

type vectorIndex struct {
	Version   int            `json:"version"`
	CreatedAt time.Time      `json:"created_at"`
	Checksum  string         `json:"checksum"`
	Chunks    []storedVector `json:"chunks"`
}

type storedVector struct {
	ID     string    `json:"id"`
	Source string    `json:"source"`
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

const indexVersion = 1

func persistIndex(path string, checksum string, chunks []vectorChunk) error {
	if path == "" {
		return nil
	}
	dir := filepath.Dir(filepath.Clean(path))
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	stored := make([]storedVector, 0, len(chunks))
	for _, chunk := range chunks {
		stored = append(stored, storedVector{
			ID:     chunk.id,
			Source: chunk.source,
			Text:   chunk.text,
			Vector: chunk.vector,
		})
	}

	index := vectorIndex{
		Version:   indexVersion,
		CreatedAt: time.Now().UTC(),
		Checksum:  checksum,
		Chunks:    stored,
	}

	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func loadIndex(path, checksum string) ([]vectorChunk, error) {
	if path == "" {
		return nil, errors.New("rag: no index path provided")
	}
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}

	var index vectorIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, err
	}
	if index.Version != indexVersion {
		return nil, fmt.Errorf("rag: index version mismatch: %d", index.Version)
	}
	if index.Checksum != checksum {
		return nil, errors.New("rag: index checksum mismatch")
	}

	chunks := make([]vectorChunk, 0, len(index.Chunks))
	for _, item := range index.Chunks {
		vector := make([]float32, len(item.Vector))
		copy(vector, item.Vector)
		normalizeVector(vector)
		chunks = append(chunks, vectorChunk{
			id:     item.ID,
			source: item.Source,
			text:   item.Text,
			vector: vector,
		})
	}

	return chunks, nil
}

func buildIndex(corpusPath string, chunkSize, overlap int, embedder embedding.Provider, allowedExt map[string]struct{}) ([]vectorChunk, error) {
	chunks := make([]vectorChunk, 0, 128)
	err := filepath.WalkDir(corpusPath, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !isSupportedExt(ext, allowedExt) {
			return nil
		}

		content, readErr := extractDocumentText(path, ext)
		if readErr != nil {
			return readErr
		}
		content = strings.TrimSpace(content)
		if content == "" {
			return nil
		}
		relPath, relErr := filepath.Rel(corpusPath, path)
		if relErr != nil {
			relPath = filepath.Base(path)
		}

		slices := chunkText(content, chunkSize, overlap)
		for idx, chunk := range slices {
			cleaned := strings.TrimSpace(chunk)
			if cleaned == "" {
				continue
			}
			embeddingVec, embedErr := embedWithTimeout(embedder, cleaned)
			if embedErr != nil {
				return embedErr
			}
			vector := make([]float32, len(embeddingVec))
			copy(vector, embeddingVec)
			normalizeVector(vector)
			chunks = append(chunks, vectorChunk{
				id:     fmt.Sprintf("%s#%d", relPath, idx),
				source: relPath,
				text:   cleaned,
				vector: vector,
			})
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("rag: failed to index corpus: %w", err)
	}
	return chunks, nil
}

func checksumCorpus(root string) (string, error) {
	h := sha256.New()
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			rel = filepath.Base(path)
		}
		fmt.Fprintf(h, "%s|%d|%d;", rel, info.Size(), info.ModTime().Unix())
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("rag: failed to checksum corpus: %w", err)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func embedWithTimeout(embedder embedding.Provider, text string) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	return embedder.Embed(ctx, text)
}

func extractDocumentText(path, ext string) (string, error) {
	switch ext {
	case ".pdf":
		return extractPDFText(path)
	default:
		data, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(data), nil
	}
}

func extractPDFText(path string) (string, error) {
	file, reader, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = file.Close()
	}()
	plain, err := reader.GetPlainText()
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, plain); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func isSupportedExt(ext string, allowed map[string]struct{}) bool {
	if len(allowed) == 0 {
		return false
	}
	normalized := strings.TrimSpace(strings.ToLower(ext))
	if normalized == "" {
		return false
	}
	if !strings.HasPrefix(normalized, ".") {
		normalized = "." + normalized
	}
	_, ok := allowed[normalized]
	return ok
}

func normalizeExtensions(list []string) map[string]struct{} {
	set := make(map[string]struct{}, len(list))
	for _, ext := range list {
		trimmed := strings.TrimSpace(strings.ToLower(ext))
		if trimmed == "" {
			continue
		}
		if !strings.HasPrefix(trimmed, ".") {
			trimmed = "." + trimmed
		}
		set[trimmed] = struct{}{}
	}
	return set
}

func chunkText(input string, size, overlap int) []string {
	words := strings.Fields(input)
	if len(words) == 0 {
		return nil
	}
	if size <= 0 {
		size = 512
	}
	step := size - overlap
	if step <= 0 {
		step = size
	}
	chunks := make([]string, 0, (len(words)/step)+1)
	for start := 0; start < len(words); start += step {
		end := start + size
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, strings.TrimSpace(chunk))
		if end == len(words) {
			break
		}
	}
	return chunks
}
