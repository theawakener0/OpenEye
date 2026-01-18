package memory

import (
	"context"
	"log"
	"sync"
	"time"
)

// AsyncWriter provides asynchronous memory persistence with buffering.
// This moves database writes off the critical path for lower latency.
type AsyncWriter struct {
	store         *VectorStore
	embedder      EmbeddingProvider
	writeQueue    chan writeRequest
	batchSize     int
	flushInterval time.Duration
	wg            sync.WaitGroup
	stopCh        chan struct{}
	mu            sync.Mutex
	running       bool
}

type writeRequest struct {
	text      string
	role      string
	embedding []float32
	timestamp time.Time
}

// AsyncWriterConfig configures the async writer.
type AsyncWriterConfig struct {
	QueueSize     int
	BatchSize     int
	FlushInterval time.Duration
}

// DefaultAsyncWriterConfig returns sensible defaults.
func DefaultAsyncWriterConfig() AsyncWriterConfig {
	return AsyncWriterConfig{
		QueueSize:     100,
		BatchSize:     10,
		FlushInterval: 500 * time.Millisecond,
	}
}

// NewAsyncWriter creates a new async writer.
func NewAsyncWriter(store *VectorStore, embedder EmbeddingProvider, cfg AsyncWriterConfig) *AsyncWriter {
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 100
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 500 * time.Millisecond
	}

	return &AsyncWriter{
		store:         store,
		embedder:      embedder,
		writeQueue:    make(chan writeRequest, cfg.QueueSize),
		batchSize:     cfg.BatchSize,
		flushInterval: cfg.FlushInterval,
		stopCh:        make(chan struct{}),
	}
}

// Start begins the async writer goroutine.
func (w *AsyncWriter) Start() {
	w.mu.Lock()
	if w.running {
		w.mu.Unlock()
		return
	}
	w.running = true
	w.mu.Unlock()

	w.wg.Add(1)
	go w.processLoop()
}

// Stop gracefully shuts down the async writer, flushing pending writes.
func (w *AsyncWriter) Stop() error {
	w.mu.Lock()
	if !w.running {
		w.mu.Unlock()
		return nil
	}
	w.running = false
	w.mu.Unlock()

	close(w.stopCh)
	w.wg.Wait()
	return nil
}

// Write queues a write request for async processing.
// Returns immediately without blocking on database operations.
func (w *AsyncWriter) Write(ctx context.Context, text, role string) error {
	if w == nil || w.store == nil {
		return nil
	}

	// Generate embedding if embedder is available
	var embedding []float32
	if w.embedder != nil {
		var err error
		embedding, err = w.embedder.Embed(ctx, text)
		if err != nil {
			log.Printf("async_writer: failed to generate embedding: %v", err)
			// Continue without embedding
		}
	}

	select {
	case w.writeQueue <- writeRequest{
		text:      text,
		role:      role,
		embedding: embedding,
		timestamp: time.Now(),
	}:
		return nil
	default:
		// Queue full - write synchronously to avoid data loss
		log.Printf("async_writer: queue full, writing synchronously")
		_, err := w.store.InsertMemory(ctx, text, "", role, embedding)
		return err
	}
}

// WriteWithEmbedding queues a write with a pre-computed embedding.
func (w *AsyncWriter) WriteWithEmbedding(text, role string, embedding []float32) error {
	if w == nil || w.store == nil {
		return nil
	}

	select {
	case w.writeQueue <- writeRequest{
		text:      text,
		role:      role,
		embedding: embedding,
		timestamp: time.Now(),
	}:
		return nil
	default:
		// Queue full - write synchronously
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := w.store.InsertMemory(ctx, text, "", role, embedding)
		return err
	}
}

// Flush forces an immediate flush of all pending writes.
func (w *AsyncWriter) Flush(ctx context.Context) error {
	// Drain the queue
	batch := make([]writeRequest, 0, w.batchSize)

	for {
		select {
		case req := <-w.writeQueue:
			batch = append(batch, req)
			if len(batch) >= w.batchSize {
				if err := w.writeBatch(ctx, batch); err != nil {
					log.Printf("async_writer: flush batch error: %v", err)
				}
				batch = batch[:0]
			}
		default:
			// Queue empty
			if len(batch) > 0 {
				return w.writeBatch(ctx, batch)
			}
			return nil
		}
	}
}

// QueueLength returns the current queue size.
func (w *AsyncWriter) QueueLength() int {
	return len(w.writeQueue)
}

func (w *AsyncWriter) processLoop() {
	defer w.wg.Done()

	batch := make([]writeRequest, 0, w.batchSize)
	ticker := time.NewTicker(w.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-w.stopCh:
			// Flush remaining items before exit
			for {
				select {
				case req := <-w.writeQueue:
					batch = append(batch, req)
				default:
					if len(batch) > 0 {
						ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
						w.writeBatch(ctx, batch)
						cancel()
					}
					return
				}
			}

		case req := <-w.writeQueue:
			batch = append(batch, req)
			if len(batch) >= w.batchSize {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				if err := w.writeBatch(ctx, batch); err != nil {
					log.Printf("async_writer: batch write error: %v", err)
				}
				cancel()
				batch = batch[:0]
			}

		case <-ticker.C:
			if len(batch) > 0 {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				if err := w.writeBatch(ctx, batch); err != nil {
					log.Printf("async_writer: flush error: %v", err)
				}
				cancel()
				batch = batch[:0]
			}
		}
	}
}

func (w *AsyncWriter) writeBatch(ctx context.Context, batch []writeRequest) error {
	for _, req := range batch {
		_, err := w.store.InsertMemory(ctx, req.text, "", req.role, req.embedding)
		if err != nil {
			log.Printf("async_writer: insert error: %v", err)
			// Continue with other items
		}
	}
	return nil
}
