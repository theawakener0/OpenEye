package context

import (
	"fmt"
	"log"
	"os"
	"strings"
)

// HistoryItem represents a single entry in the conversational memory.
type HistoryItem struct {
	Role    string
	Content string
}

type Context struct {
	SysMsg       string
	prompt       string
	history      []HistoryItem
	templatePath string
	summary      string
	knowledge    []string

	// MaxHistoryChars caps the character budget for conversation history.
	// 0 uses the default of 8000 characters. Set lower for edge devices
	// with small context windows (e.g., 2048 tokens ~ 5000 chars).
	MaxHistoryChars int
}

func NewContext(sysMsg string, prompt string) *Context {
	return &Context{
		SysMsg: sysMsg,
		prompt: prompt,
	}
}

func (c *Context) GetPrompt() string {
	if c.prompt != "" {
		return c.prompt
	}
	envPrompt := os.Getenv("APP_PROMPT")
	if envPrompt != "" {
		return envPrompt
	}
	defaultPrompt := "Hello, world!"
	log.Printf("Using default prompt: %s", defaultPrompt)
	return defaultPrompt
}

func (c *Context) SetPrompt(prompt string) {
	c.prompt = prompt
}

func (c *Context) SetHistory(history []HistoryItem) {
	c.history = append([]HistoryItem(nil), history...)
}

func (c *Context) History() []HistoryItem {
	return append([]HistoryItem(nil), c.history...)
}

// SetSummary stores a condensed view of the memory when available.
func (c *Context) SetSummary(summary string) {
	c.summary = strings.TrimSpace(summary)
}

// Summary returns the current memory summary, if any.
func (c *Context) Summary() string {
	return c.summary
}

// SetKnowledge hydrates retrieved context blocks used for RAG prompts.
func (c *Context) SetKnowledge(items []string) {
	c.knowledge = append([]string(nil), items...)
}

// Knowledge returns a copy of the retrieved context snippets.
func (c *Context) Knowledge() []string {
	return append([]string(nil), c.knowledge...)
}

// SetTemplatePath overrides the context template file path if provided.
func (c *Context) SetTemplatePath(path string) {
	c.templatePath = path
}

func (c *Context) GetSysMsg() string {
	if c.SysMsg != "" {
		return c.SysMsg
	}

	envSysMsg := os.Getenv("APP_SYSMSG")
	if envSysMsg != "" {
		return envSysMsg
	}

	defaultSysMsg := "You are a helpful AI assistant called OpenEye. Answer the user's questions directly and helpfully. Be concise, factual, and direct in your responses."

	//	log.Printf("Using default system message: %s", defaultSysMsg)
	return defaultSysMsg
}

func (c *Context) SetSysMsg(sysMsg string) {
	c.SysMsg = sysMsg
}

func (c *Context) Format() string {
	// Use ChatML format for chat models
	return c.FormatChatML()
}

// FormatChatML formats the context using a smart token budget.
func (c *Context) FormatChatML() string {
	var b strings.Builder

	// 1. Build immutable system section
	var sysBlock strings.Builder
	sysBlock.WriteString("<|im_start|>\n<|system|>\n")
	sysBlock.WriteString(c.GetSysMsg())

	if c.summary != "" {
		sysBlock.WriteString("\n\n<|Memory Summary|>\n")
		sysBlock.WriteString(c.summary)
	}

	if len(c.knowledge) > 0 {
		sysBlock.WriteString("\n\n<|Retrieved Context|>\n")
		for idx, block := range c.knowledge {
			if block == "" {
				continue
			}
			sysBlock.WriteString(fmt.Sprintf("[%d] %s\n", idx+1, block))
		}
	}
	sysBlock.WriteString("<|im_end|>\n")

	// 2. Build immutable current prompt
	promptBlock := fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", c.GetPrompt())

	// 3. Calculate remaining budget for history
	// Assuming a safe default context window if we can't measure exactly.
	// In a real system, we'd use a tokenizer. Here we estimate 1 char ~= 0.3 tokens approx, or just use char limits.
	maxHistoryChars := c.MaxHistoryChars
	if maxHistoryChars <= 0 {
		maxHistoryChars = 8000 // default for typical models
	}

	// 4. Build history section backwards until full
	var historyBlock string
	currentChars := 0

	// Iterate backwards
	for i := len(c.history) - 1; i >= 0; i-- {
		item := c.history[i]
		if item.Role == "" && item.Content == "" {
			continue
		}
		role := strings.ToLower(item.Role)
		if role == "" {
			role = "user"
		}

		entry := fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", role, item.Content)
		if currentChars+len(entry) > maxHistoryChars {
			break
		}

		// Prepend since we are iterating backwards
		historyBlock = entry + historyBlock
		currentChars += len(entry)
	}

	b.WriteString(sysBlock.String())
	b.WriteString(historyBlock)
	b.WriteString(promptBlock)

	return b.String()
}

func (c *Context) GetFormat() string {
	path := c.templatePath
	if path == "" {
		path = os.Getenv("APP_CONTEXT_PATH")
	}
	if path == "" {
		return c.Format()
	}

	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("Error reading context format file %q: %v", path, err)
		return c.Format()
	}

	return string(data)
}
