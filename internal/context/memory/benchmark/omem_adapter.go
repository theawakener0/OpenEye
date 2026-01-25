package benchmark

import (
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
	"context"
	"fmt"
	"math"
	"strings"
)

// OmemAdapter implements MemorySystemAdapter for the Omem system.
type OmemAdapter struct {
	cfg      omem.Config
	engine   *omem.Engine
	manager  *runtime.Manager
	embedder embedding.Provider
	preset   omem.AblationPreset
}

func NewOmemAdapter(preset omem.AblationPreset, manager *runtime.Manager, embedder embedding.Provider) *OmemAdapter {
	cfg := omem.AblationConfig(preset)
	// Use a unique DB for benchmarking to avoid interference
	cfg.Storage.DBPath = fmt.Sprintf("benchmark_omem_%s.db", string(preset))
	return &OmemAdapter{
		cfg:      cfg,
		manager:  manager,
		embedder: embedder,
		preset:   preset,
	}
}

func (a *OmemAdapter) Name() string {
	return fmt.Sprintf("omem_%s", a.preset)
}

func (a *OmemAdapter) Initialize(ctx context.Context) error {
	var err error
	a.engine, err = omem.NewEngine(a.cfg)
	if err != nil {
		return err
	}

	// Define LLM and Embedding functions for the engine
	llmFunc := func(ctx context.Context, prompt string) (string, error) {
		// For benchmarking, we use the manager if available, or a mock
		if a.manager != nil {
			resp, err := a.manager.Generate(ctx, runtime.Request{
				Prompt: prompt,
			})
			if err != nil {
				return "", err
			}
			return resp.Text, nil
		}
		return "Mock LLM response for benchmarking", nil
	}

	embFunc := func(ctx context.Context, text string) ([]float32, error) {
		if a.embedder != nil {
			return a.embedder.Embed(ctx, text)
		}

		// Clean the text slightly for more robust matching
		text = strings.ToLower(strings.TrimSpace(text))

		// For benchmarking without a real embedder, generate a deterministic mock
		// so that retrieval (even if based on random-ish vectors) is consistent.
		dim := 384
		vec := make([]float32, dim)
		// Simple hash of text to fill vector
		var h uint32 = 0x811c9dc5
		for i := 0; i < len(text); i++ {
			h ^= uint32(text[i])
			h *= 0x01000193
		}
		for i := 0; i < dim; i++ {
			h ^= uint32(i)
			h *= 0x01000193
			vec[i] = float32(h) / float32(math.MaxUint32)
		}
		// Normalize to ensure high cosine similarity for exact matches
		var norm float32
		for _, v := range vec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		for i := range vec {
			vec[i] /= norm
		}
		return vec, nil
	}

	return a.engine.Initialize(llmFunc, embFunc)
}

func (a *OmemAdapter) Store(ctx context.Context, role, content string, turnIndex int) error {
	if a.engine == nil {
		return fmt.Errorf("engine not initialized")
	}
	_, err := a.engine.ProcessText(ctx, content, role)
	return err
}

func (a *OmemAdapter) Retrieve(ctx context.Context, query string, limit int) ([]string, error) {
	if a.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}
	facts, err := a.engine.GetFacts(ctx, query)
	if err != nil {
		return nil, err
	}

	var results []string
	for i, f := range facts {
		if i >= limit {
			break
		}
		results = append(results, f.Fact.AtomicText)
	}
	return results, nil
}

func (a *OmemAdapter) BuildContext(ctx context.Context, query string, maxTokens int) (string, int, error) {
	if a.engine == nil {
		return "", 0, fmt.Errorf("engine not initialized")
	}
	res, err := a.engine.GetContextForPrompt(ctx, query, maxTokens)
	if err != nil {
		return "", 0, err
	}
	return res.FormattedContext, res.TokenEstimate, nil
}

func (a *OmemAdapter) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if a.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}
	return a.engine.GetStats(ctx), nil
}

func (a *OmemAdapter) Close() error {
	if a.engine != nil {
		return a.engine.Close()
	}
	return nil
}
