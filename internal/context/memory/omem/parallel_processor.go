package omem

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ParallelProcessor provides goroutine pools for efficient batch processing.
// Key features:
// - Bounded worker pool to prevent resource exhaustion
// - Async queue for non-blocking operations
// - Batch processing with configurable chunk size
// - Graceful shutdown with timeout
type ParallelProcessor struct {
	config      ParallelConfig
	workerCount int

	// Worker pool
	taskCh   chan processorTask
	resultCh chan processorResult
	wg       sync.WaitGroup
	running  atomic.Bool

	// Statistics
	taskCount      atomic.Int64
	completedCount atomic.Int64
	errorCount     atomic.Int64

	// Shutdown
	stopCh    chan struct{}
	stoppedCh chan struct{}
}

// processorTask represents a unit of work.
type processorTask struct {
	id       int64
	fn       func(ctx context.Context) error
	ctx      context.Context
	resultCh chan<- processorResult
}

// processorResult represents the result of a task.
type processorResult struct {
	id  int64
	err error
}

// BatchTask represents a task in a batch operation.
type BatchTask[T any, R any] struct {
	Input  T
	Output R
	Error  error
	Index  int
}

// NewParallelProcessor creates a new parallel processor.
func NewParallelProcessor(cfg ParallelConfig) *ParallelProcessor {
	cfg = applyParallelDefaults(cfg)

	workerCount := cfg.MaxWorkers
	if workerCount <= 0 {
		workerCount = runtime.NumCPU()
	}

	pp := &ParallelProcessor{
		config:      cfg,
		workerCount: workerCount,
		taskCh:      make(chan processorTask, cfg.QueueSize),
		resultCh:    make(chan processorResult, cfg.QueueSize),
		stopCh:      make(chan struct{}),
		stoppedCh:   make(chan struct{}),
	}

	return pp
}

// applyParallelDefaults fills in missing configuration values.
func applyParallelDefaults(cfg ParallelConfig) ParallelConfig {
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = runtime.NumCPU()
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 100
	}
	return cfg
}

// Start starts the worker pool.
func (pp *ParallelProcessor) Start() {
	if pp == nil || pp.running.Swap(true) {
		return // Already running
	}

	// Start workers
	for i := 0; i < pp.workerCount; i++ {
		pp.wg.Add(1)
		go pp.worker(i)
	}
}

// worker processes tasks from the queue.
func (pp *ParallelProcessor) worker(id int) {
	defer pp.wg.Done()

	for {
		select {
		case <-pp.stopCh:
			return
		case task, ok := <-pp.taskCh:
			if !ok {
				return
			}

			// Execute the task
			err := task.fn(task.ctx)

			pp.completedCount.Add(1)
			if err != nil {
				pp.errorCount.Add(1)
			}

			// Send result if channel provided
			if task.resultCh != nil {
				select {
				case task.resultCh <- processorResult{id: task.id, err: err}:
				default:
					// Result channel full or closed, drop result
				}
			}
		}
	}
}

// Submit submits a task for async processing.
func (pp *ParallelProcessor) Submit(ctx context.Context, fn func(ctx context.Context) error) error {
	if pp == nil {
		return fn(ctx) // Run synchronously if no processor
	}

	if !pp.running.Load() {
		pp.Start()
	}

	taskID := pp.taskCount.Add(1)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-pp.stopCh:
		return ErrProcessorStopped
	case pp.taskCh <- processorTask{
		id:  taskID,
		fn:  fn,
		ctx: ctx,
	}:
		return nil
	}
}

// SubmitAndWait submits a task and waits for completion.
func (pp *ParallelProcessor) SubmitAndWait(ctx context.Context, fn func(ctx context.Context) error) error {
	if pp == nil {
		return fn(ctx)
	}

	if !pp.running.Load() {
		pp.Start()
	}

	taskID := pp.taskCount.Add(1)
	resultCh := make(chan processorResult, 1)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-pp.stopCh:
		return ErrProcessorStopped
	case pp.taskCh <- processorTask{
		id:       taskID,
		fn:       fn,
		ctx:      ctx,
		resultCh: resultCh,
	}:
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case result := <-resultCh:
		return result.err
	}
}

// ProcessBatch processes a batch of items in parallel.
func ProcessBatch[T any, R any](
	ctx context.Context,
	pp *ParallelProcessor,
	items []T,
	processor func(ctx context.Context, item T) (R, error),
) []BatchTask[T, R] {
	if len(items) == 0 {
		return nil
	}

	results := make([]BatchTask[T, R], len(items))
	var wg sync.WaitGroup
	var mu sync.Mutex

	// Determine concurrency
	concurrency := runtime.NumCPU()
	if pp != nil {
		concurrency = pp.workerCount
	}

	// Create work channel
	workCh := make(chan int, len(items))
	for i := range items {
		workCh <- i
	}
	close(workCh)

	// Limit workers to item count
	if concurrency > len(items) {
		concurrency = len(items)
	}

	// Start workers
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range workCh {
				select {
				case <-ctx.Done():
					return
				default:
				}

				output, err := processor(ctx, items[idx])

				mu.Lock()
				results[idx] = BatchTask[T, R]{
					Input:  items[idx],
					Output: output,
					Error:  err,
					Index:  idx,
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	return results
}

// ProcessBatchWithSize processes items in batches of specified size.
func ProcessBatchWithSize[T any, R any](
	ctx context.Context,
	pp *ParallelProcessor,
	items []T,
	batchSize int,
	processor func(ctx context.Context, batch []T) ([]R, error),
) ([]R, error) {
	if len(items) == 0 {
		return nil, nil
	}

	if batchSize <= 0 {
		if pp != nil {
			batchSize = pp.config.BatchSize
		} else {
			batchSize = 10
		}
	}

	// Split into batches
	var batches [][]T
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batches = append(batches, items[i:end])
	}

	// Process batches in parallel
	batchResults := ProcessBatch(ctx, pp, batches, processor)

	// Combine results
	var allResults []R
	for _, br := range batchResults {
		if br.Error != nil {
			return allResults, br.Error
		}
		allResults = append(allResults, br.Output...)
	}

	return allResults, nil
}

// Stop gracefully stops the processor with a timeout.
func (pp *ParallelProcessor) Stop(timeout time.Duration) error {
	if pp == nil || !pp.running.Swap(false) {
		return nil // Not running
	}

	close(pp.stopCh)

	// Wait with timeout
	done := make(chan struct{})
	go func() {
		pp.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		close(pp.stoppedCh)
		return nil
	case <-time.After(timeout):
		return ErrShutdownTimeout
	}
}

// QueueLength returns the current number of queued tasks.
func (pp *ParallelProcessor) QueueLength() int {
	if pp == nil {
		return 0
	}
	return len(pp.taskCh)
}

// GetStats returns processor statistics.
func (pp *ParallelProcessor) GetStats() map[string]interface{} {
	if pp == nil {
		return nil
	}

	return map[string]interface{}{
		"worker_count":    pp.workerCount,
		"running":         pp.running.Load(),
		"queue_length":    len(pp.taskCh),
		"total_tasks":     pp.taskCount.Load(),
		"completed_tasks": pp.completedCount.Load(),
		"error_count":     pp.errorCount.Load(),
		"config": map[string]interface{}{
			"max_workers": pp.config.MaxWorkers,
			"batch_size":  pp.config.BatchSize,
			"queue_size":  pp.config.QueueSize,
		},
	}
}

// ============================================================================
// Async Task Helpers
// ============================================================================

// AsyncResult represents the result of an async operation.
type AsyncResult[T any] struct {
	Value T
	Error error
}

// Future represents a pending async result.
type Future[T any] struct {
	ch   chan AsyncResult[T]
	once sync.Once
}

// RunAsync executes a function asynchronously and returns a future.
func RunAsync[T any](ctx context.Context, fn func(ctx context.Context) (T, error)) *Future[T] {
	f := &Future[T]{
		ch: make(chan AsyncResult[T], 1),
	}

	go func() {
		var result AsyncResult[T]
		result.Value, result.Error = fn(ctx)

		select {
		case f.ch <- result:
		default:
		}
	}()

	return f
}

// Get waits for and returns the result.
func (f *Future[T]) Get(ctx context.Context) (T, error) {
	select {
	case <-ctx.Done():
		var zero T
		return zero, ctx.Err()
	case result := <-f.ch:
		// Put result back for repeated calls
		select {
		case f.ch <- result:
		default:
		}
		return result.Value, result.Error
	}
}

// TryGet returns the result if available, otherwise returns false.
func (f *Future[T]) TryGet() (T, bool, error) {
	select {
	case result := <-f.ch:
		// Put result back
		select {
		case f.ch <- result:
		default:
		}
		return result.Value, true, result.Error
	default:
		var zero T
		return zero, false, nil
	}
}

// ============================================================================
// Errors
// ============================================================================

// Common errors.
var (
	ErrProcessorStopped = parallelError("processor stopped")
	ErrShutdownTimeout  = parallelError("shutdown timeout")
)

// parallelError is a simple error type for parallel processing.
type parallelError string

func (e parallelError) Error() string { return string(e) }
