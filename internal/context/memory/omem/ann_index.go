package omem

import "context"

// ANNCandidate is a candidate result returned by an ANN index.
type ANNCandidate struct {
	FactID int64
	Score  float64
}

// VectorCandidateIndex abstracts approximate semantic retrieval.
type VectorCandidateIndex interface {
	Upsert(ctx context.Context, factID int64, embedding []float32) error
	Delete(ctx context.Context, factID int64) error
	Search(ctx context.Context, query []float32, k int, oversample int) ([]ANNCandidate, error)
	Rebuild(ctx context.Context, facts []Fact) error
	Stats(ctx context.Context) (map[string]interface{}, error)
	Close() error
}
