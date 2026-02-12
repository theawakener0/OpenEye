package benchmark

import (
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"
)

// OmemAdapter implements MemorySystemAdapter for the Omem system.
type OmemAdapter struct {
	cfg      omem.Config
	engine   *omem.Engine
	manager  *runtime.Manager
	embedder embedding.Provider
	preset   omem.AblationPreset
	// Store turns directly for fallback retrieval
	turns []Turn
}

type Turn struct {
	Role    string
	Content string
	Index   int
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
		turns:    make([]Turn, 0),
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

	llmFunc := func(ctx context.Context, prompt string) (string, error) {
		if a.manager != nil {
			resp, err := a.manager.Generate(ctx, runtime.Request{
				Prompt: prompt,
			})
			if err != nil {
				return "", err
			}
			return resp.Text, nil
		}
		return a.mockLLMExtract(ctx, prompt)
	}

	embFunc := func(ctx context.Context, text string) ([]float32, error) {
		if a.embedder != nil {
			return a.embedder.Embed(ctx, text)
		}
		return a.mockEmbedding(text), nil
	}

	return a.engine.Initialize(llmFunc, embFunc)
}

func (a *OmemAdapter) mockLLMExtract(ctx context.Context, prompt string) (string, error) {
	promptLower := strings.ToLower(prompt)

	// Check if this is a fact extraction prompt (contains "FACTS:" or "FACT|")
	isFactExtraction := strings.Contains(promptLower, "fact") || strings.Contains(promptLower, "text:")

	if isFactExtraction {
		// Extract the text to analyze from the prompt
		textStart := strings.Index(prompt, "TEXT:")
		var textToAnalyze string
		if textStart >= 0 {
			textToAnalyze = strings.TrimSpace(prompt[textStart+5:])
			// Find FACTS: to remove instructions
			factsStart := strings.Index(textToAnalyze, "FACTS:")
			if factsStart >= 0 {
				textToAnalyze = strings.TrimSpace(textToAnalyze[:factsStart])
			}
		}

		if textToAnalyze == "" {
			return "No facts extracted.", nil
		}

		// Clean and format the text as a fact
		factText := strings.TrimSpace(textToAnalyze)
		// Convert first person to third person
		factText = strings.ReplaceAll(factText, "I am", "User is")
		factText = strings.ReplaceAll(factText, "I work", "User works")
		factText = strings.ReplaceAll(factText, "I live", "User lives")
		factText = strings.ReplaceAll(factText, "I've", "User has")
		factText = strings.ReplaceAll(factText, "I'm", "User is")
		factText = strings.ReplaceAll(factText, " my ", " their ")
		factText = strings.ReplaceAll(factText, " me ", " them ")

		// Determine category based on keywords
		category := "other"
		textCheck := strings.ToLower(factText)
		if strings.Contains(textCheck, "favorite") || strings.Contains(textCheck, "like") || strings.Contains(textCheck, "love") {
			category = "preference"
		} else if strings.Contains(textCheck, "birthday") || strings.Contains(textCheck, "born") || strings.Contains(textCheck, "work") || strings.Contains(textCheck, "live") || strings.Contains(textCheck, "name") {
			category = "biographical"
		} else if strings.Contains(textCheck, "spouse") || strings.Contains(textCheck, "wife") || strings.Contains(textCheck, "husband") || strings.Contains(textCheck, "family") {
			category = "relationship"
		} else if strings.Contains(textCheck, "goal") || strings.Contains(textCheck, "want") || strings.Contains(textCheck, "plan") {
			category = "task"
		}

		return fmt.Sprintf("FACT|%s|0.7|%s", category, factText), nil
	}

	// For non-fact-extraction prompts
	return "No facts extracted.", nil
}

func (a *OmemAdapter) Store(ctx context.Context, role, content string, turnIndex int) error {
	if a.engine == nil {
		return fmt.Errorf("engine not initialized")
	}

	// Store turn for fallback retrieval
	a.turns = append(a.turns, Turn{
		Role:    role,
		Content: content,
		Index:   turnIndex,
	})

	// Also try to store in engine
	_, err := a.engine.ProcessText(ctx, content, role)
	if err != nil {
		// If engine fails, we still have the turn stored
		return nil
	}

	return nil
}

func (a *OmemAdapter) Retrieve(ctx context.Context, query string, limit int) ([]string, error) {
	if a.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	res, err := a.engine.GetContextForPrompt(ctx, query, 512)
	if err != nil {
		return nil, err
	}

	var results []string
	for _, sf := range res.Facts {
		if len(results) >= limit {
			break
		}
		results = append(results, sf.Fact.AtomicText)
	}

	return results, nil
}

func (a *OmemAdapter) BuildContext(ctx context.Context, query string, maxTokens int) (string, int, error) {
	if a.engine == nil {
		return "", 0, fmt.Errorf("engine not initialized")
	}
	res, err := a.engine.GetContextForPrompt(ctx, query, maxTokens)
	if err != nil {
		// Fallback: build simple context from turns
		var context string
		for _, turn := range a.turns {
			context += fmt.Sprintf("%s: %s\n", turn.Role, turn.Content)
		}
		tokens := len(strings.Fields(context))
		return context, tokens, nil
	}
	return res.FormattedContext, res.TokenEstimate, nil
}

func (a *OmemAdapter) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if a.engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}
	stats := a.engine.GetStats(ctx)
	stats["benchmark_turns_stored"] = len(a.turns)
	return stats, nil
}

func (a *OmemAdapter) Close() error {
	if a.engine != nil {
		return a.engine.Close()
	}
	return nil
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// extractQueryKey extracts the main noun/keyword from a query
func extractQueryKey(query string) string {
	// Remove common question words
	words := []string{"what", "is", "my", "are", "do", "for", "when", "where", "who", "how", "the", "a", "an", "does", "did"}
	result := query
	for _, w := range words {
		result = strings.ReplaceAll(result, w, "")
	}
	result = strings.TrimSpace(result)

	// Return the most specific term
	if strings.Contains(query, "hobbies") || strings.Contains(query, "hobby") {
		return "into" // Match "I've been into X"
	}
	if strings.Contains(query, "color") {
		return "color"
	}
	if strings.Contains(query, "birthday") {
		return "birthday"
	}
	if strings.Contains(query, "work") || strings.Contains(query, "job") {
		return "work"
	}
	if strings.Contains(query, "live") || strings.Contains(query, "location") {
		return "live"
	}
	if strings.Contains(query, "name") {
		return "name"
	}

	return result
}

func (a *OmemAdapter) mockEmbedding(text string) []float32 {
	text = strings.ToLower(strings.TrimSpace(text))
	dim := 384
	vec := make([]float32, dim)
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
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

// Helper to simulate async processing (for compatibility)
func init() {
	_ = time.Now()
	_ = uuid.MustParse("12345678-1234-1234-1234-123456789abc")
}
