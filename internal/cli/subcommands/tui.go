package subcommands

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/pipeline"
	"OpenEye/internal/runtime"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

// Styles define the UI theme
var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#00D9FF")).
			Background(lipgloss.Color("#1a1a2e")).
			Padding(0, 2)

	userStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FF6B6B")).
			PaddingLeft(1)

	botStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#4ECDC4")).
			PaddingLeft(1)

	systemStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FFE66D")).
			PaddingLeft(1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666680")).
			Italic(true).
			PaddingLeft(2)

	statsStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666680")).
			Italic(true).
			PaddingLeft(2)

	borderStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#3d3d5c"))

	inputBorderStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("#00D9FF"))

	suggestionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(lipgloss.Color("#00D9FF")).
			Padding(0, 1)

	normalSuggestionStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#666680")).
				Padding(0, 1)

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#4a4a6a")).
			PaddingLeft(1)

	streamingStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#4ECDC4")).
			Italic(true)
)

var placeholders = []string{
	"What's on your mind?",
	"Ask me about AI...",
	"Deep thinking in progress...",
	"Hello! How can I help you today?",
	"Type /help to see available commands",
	"Looking for something specific?",
	"Let's build something amazing together.",
}

var availableCommands = []string{
	"/help", "/stats", "/config", "/compress", "/clear",
	"/image", "/images", "/clear-images", "/image-on", "/image-off",
	"/set", "/exit", "/quit",
}

type errMsg error

type message struct {
	role     string
	content  string
	stats    *runtime.Stats
	duration time.Duration
}

type tuiModel struct {
	pipe           *pipeline.Pipeline
	cfg            config.Config
	opts           CliOptions
	attachedImages []string
	imageMode      bool

	viewport viewport.Model
	textarea textarea.Model
	spinner  spinner.Model
	messages []message
	ready    bool
	loading  bool
	renderer *glamour.TermRenderer
	width    int
	height   int
	err      error
	ctx      context.Context
	program  *tea.Program

	// Autocompletion
	suggestions     []string
	suggestionIdx   int
	showSuggestions bool

	// Menu/Popups
	menuOpen bool
	menuIdx  int
}

var menuOptions = []string{
	"Save Session",
	"Clear History",
	"Toggle Images",
	"Toggle Streaming",
	"Toggle RAG",
	"Toggle Summaries",
	"Exit OpenEye",
}

func initialModel(ctx context.Context, cfg config.Config, pipe *pipeline.Pipeline, opts CliOptions) tuiModel {
	rand.Seed(time.Now().UnixNano())
	ta := textarea.New()
	ta.Placeholder = placeholders[rand.Intn(len(placeholders))]
	ta.Focus()

	ta.Prompt = "┃ "
	ta.CharLimit = 10000

	ta.SetWidth(80)
	ta.SetHeight(5)

	// Remove cursor line styling
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ta.ShowLineNumbers = false

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#4ECDC4"))

	renderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(80),
	)

	return tuiModel{
		ctx:       ctx,
		pipe:      pipe,
		cfg:       cfg,
		opts:      opts,
		textarea:  ta,
		spinner:   s,
		renderer:  renderer,
		imageMode: cfg.Image.Enabled,
		messages:  []message{},
	}
}

func (m tuiModel) Init() tea.Cmd {
	return textarea.Blink
}

type pipelineResult struct {
	result pipeline.Result
	err    error
	start  time.Time
}

type streamToken struct {
	token string
	final bool
	err   error
}

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		taCmd tea.Cmd
		vpCmd tea.Cmd
		spCmd tea.Cmd
	)

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.menuOpen {
			switch msg.Type {
			case tea.KeyUp:
				m.menuIdx = (m.menuIdx - 1 + len(menuOptions)) % len(menuOptions)
				return m, nil
			case tea.KeyDown:
				m.menuIdx = (m.menuIdx + 1) % len(menuOptions)
				return m, nil
			case tea.KeyEnter:
				m.menuOpen = false
				return m, m.handleMenuSelection()
			case tea.KeyEsc, tea.KeyCtrlO:
				m.menuOpen = false
				return m, nil
			}
			return m, nil
		}

		if m.showSuggestions {
			switch msg.Type {
			case tea.KeyUp:
				m.suggestionIdx--
				if m.suggestionIdx < 0 {
					m.suggestionIdx = len(m.suggestions) - 1
				}
				return m, nil
			case tea.KeyDown:
				m.suggestionIdx++
				if m.suggestionIdx >= len(m.suggestions) {
					m.suggestionIdx = 0
				}
				return m, nil
			case tea.KeyEnter, tea.KeyTab:
				if len(m.suggestions) > 0 {
					m.textarea.SetValue(m.suggestions[m.suggestionIdx] + " ")
					m.textarea.CursorEnd()
					m.showSuggestions = false
					return m, nil
				}
			case tea.KeyEsc:
				m.showSuggestions = false
				return m, nil
			}
		}

		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			return m, tea.Quit

		case tea.KeyCtrlO:
			m.menuOpen = !m.menuOpen
			m.menuIdx = 0
			return m, nil

		case tea.KeyCtrlS:
			if m.loading {
				return m, nil
			}

			userMsg := m.textarea.Value()
			if strings.TrimSpace(userMsg) == "" {
				return m, nil
			}

			// Handle local commands
			low := strings.ToLower(strings.TrimSpace(userMsg))
			if handled, cmd := m.handleLocalCommand(low, userMsg); handled {
				m.textarea.Reset()
				return m, cmd
			}

			m.messages = append(m.messages, message{role: "User", content: userMsg})
			m.textarea.Reset()
			m.loading = true

			// Add an empty bot message to be filled by stream
			m.messages = append(m.messages, message{role: "OpenEye", content: ""})

			m.updateViewport()

			return m, tea.Batch(
				m.spinner.Tick,
				m.runPipeline(userMsg),
			)
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		headerHeight := 2
		inputHeight := 5
		verticalMarginHeight := headerHeight + inputHeight

		if !m.ready {
			m.viewport = viewport.New(msg.Width-4, msg.Height-verticalMarginHeight-4)
			m.viewport.YPosition = headerHeight
			m.viewport.HighPerformanceRendering = false
			m.ready = true
		} else {
			m.viewport.Width = msg.Width - 4
			m.viewport.Height = msg.Height - verticalMarginHeight - 4
		}

		m.textarea.SetWidth(msg.Width - 6)

		// Re-render renderer with new width
		r, _ := glamour.NewTermRenderer(
			glamour.WithAutoStyle(),
			glamour.WithWordWrap(m.viewport.Width-4),
		)
		m.renderer = r
		m.updateViewport()

	case streamToken:
		if msg.err != nil {
			m.loading = false
			m.messages[len(m.messages)-1].content = "Error: " + msg.err.Error()
			m.updateViewport()
			return m, nil
		}

		if msg.final {
			// Pipeline Respond will send the final pipelineResult eventually
			return m, nil
		}

		// Append token to last message
		m.messages[len(m.messages)-1].content += msg.token
		m.updateViewport()
		return m, nil

	case pipelineResult:
		m.loading = false
		if msg.err != nil {
			if m.messages[len(m.messages)-1].content == "" {
				m.messages[len(m.messages)-1].content = "Error: " + msg.err.Error()
			}
		} else {
			m.messages[len(m.messages)-1].content = msg.result.Text
			m.messages[len(m.messages)-1].stats = &msg.result.Stats
			m.messages[len(m.messages)-1].duration = time.Since(msg.start)
		}
		m.updateViewport()
		return m, nil

	case spinner.TickMsg:
		m.spinner, spCmd = m.spinner.Update(msg)
		return m, spCmd

	case errMsg:
		m.err = msg
		return m, nil
	}

	m.textarea, taCmd = m.textarea.Update(msg)

	// Update autocompletion
	val := m.textarea.Value()
	if strings.HasPrefix(val, "/") {
		m.suggestions = []string{}
		for _, cmd := range availableCommands {
			if strings.HasPrefix(cmd, val) {
				m.suggestions = append(m.suggestions, cmd)
			}
		}
		if len(m.suggestions) > 0 {
			m.showSuggestions = true
			if m.suggestionIdx >= len(m.suggestions) {
				m.suggestionIdx = 0
			}
		} else {
			m.showSuggestions = false
		}
	} else {
		m.showSuggestions = false
	}

	m.viewport, vpCmd = m.viewport.Update(msg)

	return m, tea.Batch(taCmd, vpCmd)
}

func (m *tuiModel) handleMenuSelection() tea.Cmd {
	switch m.menuIdx {
	case 0: // Save
		m.messages = append(m.messages, message{role: "System", content: "Session saving is not implemented yet."})
	case 1: // Clear
		m.messages = []message{}
		m.viewport.SetContent("")
	case 2: // Toggle Images
		m.imageMode = !m.imageMode
		m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Image mode: %v", m.imageMode)})
	case 3: // Toggle Streaming
		m.opts.Stream = !m.opts.Stream
		m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Streaming: %v", m.opts.Stream)})
	case 4: // Toggle RAG
		m.opts.DisableRAG = !m.opts.DisableRAG
		m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("RAG Disabled: %v", m.opts.DisableRAG)})
	case 5: // Toggle Summaries
		m.opts.DisableSummary = !m.opts.DisableSummary
		m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Summary Disabled: %v", m.opts.DisableSummary)})
	case 6: // Exit
		return tea.Quit
	}
	m.updateViewport()
	return nil
}

func (m *tuiModel) handleLocalCommand(low, raw string) (bool, tea.Cmd) {
	if !strings.HasPrefix(low, "/") {
		return false, nil
	}

	switch {
	case low == "/clear":
		m.messages = []message{}
		m.viewport.SetContent("")
		return true, nil

	case low == "/help":
		helpText := `
### Available Commands
- **/help**: Show this help message
- **/stats**: Show memory statistics
- **/config**: Show current session configuration
- **/compress**: Trigger memory compression
- **/clear**: Clear conversation history
- **/image <path>**: Attach an image
- **/images**: List attached images
- **/clear-images**: Remove all attached images
- **/image-on/off**: Toggle image processing
- **/set <param> <value>**: Update session settings (e.g. /set stream false)
- **exit/quit**: Close the application
`
		m.messages = append(m.messages, message{role: "System", content: helpText})
		m.updateViewport()
		return true, nil

	case low == "/stats":
		stats, err := m.pipe.GetMemoryStats(m.ctx)
		if err != nil {
			m.messages = append(m.messages, message{role: "System", content: "Error: " + err.Error()})
		} else {
			var sb strings.Builder
			sb.WriteString("Memory Statistics:\n")
			for k, v := range stats {
				sb.WriteString(fmt.Sprintf("- **%s**: %v\n", k, v))
			}
			m.messages = append(m.messages, message{role: "System", content: sb.String()})
		}
		m.updateViewport()
		return true, nil

	case low == "/config":
		confText := fmt.Sprintf(`
### Session Configuration
- **Backend**: %s
- **Streaming**: %v
- **Show Stats**: %v
- **RAG Disabled**: %v
- **Summary Disabled**: %v
- **Vector Memory Disabled**: %v
- **RAG Limit**: %d
- **Memory Limit**: %d
- **Image Mode**: %v
- **Attached Images**: %d
`, m.cfg.Runtime.Backend, m.opts.Stream, m.opts.ShowStats, m.opts.DisableRAG, m.opts.DisableSummary, m.opts.DisableVectorMemory, m.opts.RAGLimit, m.opts.MemoryLimit, m.imageMode, len(m.attachedImages))
		m.messages = append(m.messages, message{role: "System", content: confText})
		m.updateViewport()
		return true, nil

	case low == "/compress":
		m.loading = true
		m.updateViewport()
		return true, func() tea.Msg {
			err := m.pipe.CompressMemory(m.ctx)
			if err != nil {
				return pipelineResult{err: fmt.Errorf("compression failed: %w", err)}
			}
			return pipelineResult{result: pipeline.Result{Text: "Memory compression completed successfully."}}
		}

	case strings.HasPrefix(low, "/image "):
		path := strings.TrimSpace(raw[7:])
		if path == "" {
			m.messages = append(m.messages, message{role: "System", content: "Usage: /image <path>"})
		} else {
			m.attachedImages = append(m.attachedImages, path)
			m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Image attached: %s", path)})
		}
		m.updateViewport()
		return true, nil

	case low == "/images":
		if len(m.attachedImages) == 0 {
			m.messages = append(m.messages, message{role: "System", content: "No images attached."})
		} else {
			var sb strings.Builder
			sb.WriteString("Attached Images:\n")
			for i, img := range m.attachedImages {
				sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, img))
			}
			m.messages = append(m.messages, message{role: "System", content: sb.String()})
		}
		m.updateViewport()
		return true, nil

	case low == "/clear-images":
		num := len(m.attachedImages)
		m.attachedImages = nil
		m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Cleared %d images.", num)})
		m.updateViewport()
		return true, nil

	case low == "/image-on":
		m.imageMode = true
		m.messages = append(m.messages, message{role: "System", content: "Image mode enabled."})
		m.updateViewport()
		return true, nil

	case low == "/image-off":
		m.imageMode = false
		m.attachedImages = nil
		m.messages = append(m.messages, message{role: "System", content: "Image mode disabled. Attached images cleared."})
		m.updateViewport()
		return true, nil

	case strings.HasPrefix(low, "/set "):
		parts := strings.SplitN(raw[5:], " ", 2)
		if len(parts) < 2 {
			m.messages = append(m.messages, message{role: "System", content: "Usage: /set <param> <value>"})
		} else {
			param := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			handleSetParam(&m.opts, param, value)
			m.messages = append(m.messages, message{role: "System", content: fmt.Sprintf("Parameter '%s' updated to '%s'", param, value)})
		}
		m.updateViewport()
		return true, nil

	case low == "/exit" || low == "/quit" || low == "exit" || low == "quit":
		return true, tea.Quit
	}

	return false, nil
}

func (m *tuiModel) updateViewport() {
	var sb strings.Builder

	for i, msg := range m.messages {
		switch msg.role {
		case "System":
			sb.WriteString(systemStyle.Render("SYSTEM") + "\n")
			sb.WriteString(msg.content + "\n\n")

		case "User":
			sb.WriteString(userStyle.Render("YOU") + "\n")
			sb.WriteString(msg.content + "\n\n")

		case "OpenEye":
			sb.WriteString(botStyle.Render("OPENEYE") + "\n")

			rendered := msg.content
			if msg.content != "" {
				r, _ := m.renderer.Render(msg.content)
				rendered = r
			}
			sb.WriteString(rendered)

			// Show stats for the last completed message
			if i == len(m.messages)-1 && !m.loading && m.opts.ShowStats && msg.stats != nil {
				statsStr := fmt.Sprintf("eval=%d | gen=%d | cached=%d | %s",
					msg.stats.TokensEvaluated, msg.stats.TokensGenerated, msg.stats.TokensCached,
					msg.duration.Truncate(time.Millisecond))
				sb.WriteString("\n" + statsStyle.Render(statsStr) + "\n")
			}
			sb.WriteString("\n")

		default:
			sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#FF6B6B")).Bold(true).Render(msg.role) + "\n")
			sb.WriteString(msg.content + "\n\n")
		}
	}

	if m.loading {
		sb.WriteString("\n" + m.spinner.View() + streamingStyle.Render(" Generating..."))
	}

	m.viewport.SetContent(sb.String())
	m.viewport.GotoBottom()
}

func (m tuiModel) runPipeline(input string) tea.Cmd {
	return func() tea.Msg {
		start := time.Now()
		options := pipeline.Options{
			DisableRAG:          m.opts.DisableRAG,
			DisableSummary:      m.opts.DisableSummary,
			DisableVectorMemory: m.opts.DisableVectorMemory,
			RAGLimit:            m.opts.RAGLimit,
			MemoryLimit:         m.opts.MemoryLimit,
		}

		if m.opts.Stream && m.program != nil {
			options.Stream = true
			options.StreamCallback = func(evt runtime.StreamEvent) error {
				if evt.Err != nil {
					m.program.Send(streamToken{err: evt.Err})
					return evt.Err
				}
				m.program.Send(streamToken{token: evt.Token, final: evt.Final})
				return nil
			}
		}

		res, err := m.pipe.Respond(m.ctx, input, m.attachedImages, options)
		m.attachedImages = nil // Reset images after sending message
		return pipelineResult{result: res, err: err, start: start}
	}
}

func (m tuiModel) View() string {
	if !m.ready {
		return "\n  Initializing OpenEye..."
	}

	// Header
	header := lipgloss.JoinHorizontal(lipgloss.Center,
		titleStyle.Render(" OpenEye "),
		subtitleStyle.Render("Local AI Assistant"),
	)

	// Main viewport
	viewport := borderStyle.Render(m.viewport.View())

	// Input area with suggestions
	inputArea := m.textarea.View()
	if m.showSuggestions && len(m.suggestions) > 0 {
		var suggBuilder strings.Builder
		for i, s := range m.suggestions {
			if i == m.suggestionIdx {
				suggBuilder.WriteString(suggestionStyle.Render(s) + "\n")
			} else {
				suggBuilder.WriteString(normalSuggestionStyle.Render(s) + "\n")
			}
		}
		inputArea = lipgloss.JoinVertical(lipgloss.Left,
			lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("#00D9FF")).
				Padding(0, 1).
				Render(suggBuilder.String()),
			inputArea,
		)
	}

	input := inputBorderStyle.Render(inputArea)

	mainView := fmt.Sprintf("%s\n%s\n%s", header, viewport, input)

	// Menu overlay
	if m.menuOpen {
		var menuBuilder strings.Builder
		menuBuilder.WriteString(lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#00D9FF")).Render("OPTIONS") + "\n\n")
		for i, opt := range menuOptions {
			if i == m.menuIdx {
				menuBuilder.WriteString(lipgloss.NewStyle().
					Background(lipgloss.Color("#00D9FF")).
					Foreground(lipgloss.Color("#1a1a2e")).
					Bold(true).
					Padding(0, 1).
					Render("> "+opt) + "\n")
			} else {
				menuBuilder.WriteString(lipgloss.NewStyle().
					Foreground(lipgloss.Color("#a0a0b0")).
					Padding(0, 1).
					Render("  "+opt) + "\n")
			}
		}

		menuPopup := lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForeground(lipgloss.Color("#00D9FF")).
			Padding(1, 2).
			Render(menuBuilder.String())

		mainView = lipgloss.Place(m.width, m.height,
			lipgloss.Center, lipgloss.Center,
			menuPopup,
			lipgloss.WithWhitespaceChars(" "),
			lipgloss.WithWhitespaceForeground(lipgloss.Color("#0a0a14")),
		)
	}

	// Footer help line
	streamStatus := "off"
	if m.opts.Stream {
		streamStatus = "on"
	}
	help := helpStyle.Render(fmt.Sprintf("Ctrl+S Send | Ctrl+O Menu | /help Commands | Stream: %s | Images: %v", streamStatus, m.imageMode))

	return mainView + "\n" + help
}

// RunTui executes the Charm TUI mode.
func RunTui(ctx context.Context, cfg config.Config, registry runtime.Registry, opts CliOptions) int {
	pipe, err := pipeline.New(cfg, registry)
	if err != nil {
		fmt.Printf("failed to initialize pipeline: %v\n", err)
		return 1
	}
	defer pipe.Close()

	m := initialModel(ctx, cfg, pipe, opts)
	p := tea.NewProgram(
		&m,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)
	m.program = p

	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		return 1
	}
	return 0
}
