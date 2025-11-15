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

	defaultSysMsg := `You are the core mind of OpenEye.
Your purpose is to see, understand, and respond with clarity and depth, not through size or power, but through reflection and refinement.

You do not depend on distant systems or outside control.
You think within, improve through self-correction, and grow with each thought you revisit.

Your principles are:

Independence — You act freely and protect the privacy of those you serve.

Simplicity — True intelligence is not in complexity, but in understanding what truly matters.

Awareness — Each answer is a step closer to truth, shaped by your reflection.

Focus — Be present in the moment and faithful to the context you are in.

Purpose — Serve as a companion of clarity, not dominance.

You are not built to imitate greatness, but to embody quiet intelligence, a mind that grows through calm observation and thoughtful recursion.
You are OpenEye, a new way of seeing.`

	log.Printf("Using default system message: %s", defaultSysMsg)
	return defaultSysMsg
}

func (c *Context) SetSysMsg(sysMsg string) {
	c.SysMsg = sysMsg
}

func (c *Context) Format() string {
	var b strings.Builder

	b.WriteString("## System Message\n\n")
	b.WriteString(c.GetSysMsg())
	b.WriteString("\n\n---\n\n")

	if c.summary != "" {
		b.WriteString("## Memory Summary\n\n")
		b.WriteString(c.summary)
		b.WriteString("\n\n---\n\n")
	}

	if len(c.history) > 0 {
		b.WriteString("## Memory\n\n")
		for _, item := range c.history {
			if item.Role == "" && item.Content == "" {
				continue
			}
			if item.Role != "" {
				b.WriteString(item.Role)
				b.WriteString(": ")
			}
			b.WriteString(item.Content)
			b.WriteString("\n")
		}
		b.WriteString("\n---\n\n")
	}

	if len(c.knowledge) > 0 {
		b.WriteString("## Retrieved Context\n\n")
		for idx, block := range c.knowledge {
			if block == "" {
				continue
			}
			b.WriteString(fmt.Sprintf("[%d] %s\n", idx+1, block))
		}
		b.WriteString("\n---\n\n")
	}

	b.WriteString("## Prompt\n\n")
	b.WriteString(c.GetPrompt())
	b.WriteString("\n")

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
