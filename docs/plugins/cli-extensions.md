# CLI Extensions

CLI extensions allow you to add custom subcommands to the OpenEye command-line interface. This enables custom workflows, administration tools, and integrations.

## Overview

OpenEye's CLI is built with a subcommand pattern. The CLI accepts different commands like `chat`, `serve`, and `memory`. You can add new subcommands for specialized functionality.

## CLI Architecture

```mermaid
graph TB
    subgraph Main["main.go"]
        Entry["CLI Entry Point"]
    end
    
    subgraph Subcommands
        Chat["chat"]
        Serve["serve"]
        Memory["memory"]
        Custom["custom\n← Your extension"]
    end
    
    subgraph Functions
        Parse["Parse global flags"]
        Config["Load configuration"]
        Dispatch["Dispatch to subcommands"]
    end
    
    Entry --> Parse
    Entry --> Config
    Config --> Dispatch
    Dispatch --> Chat
    Dispatch --> Serve
    Dispatch --> Memory
    Dispatch --> Custom

## Subcommand Pattern

### Basic Subcommand Structure

```go
package subcommands

import (
    "context"
    "flag"
    "fmt"
    "os"

    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

// RunCustom executes the custom subcommand
func RunCustom(ctx context.Context, cfg config.Config, registry runtime.Registry) int {
    // Parse flags
    verbose := flag.Bool("v", false, "Verbose output")
    limit := flag.Int("limit", 10, "Limit results")
    flag.Parse()
    
    // Validate arguments
    args := flag.Args()
    if len(args) < 1 {
        fmt.Fprintf(os.Stderr, "Usage: OpenEye custom [flags] <query>\n")
        return 1
    }
    
    query := args[0]
    
    if *verbose {
        fmt.Printf("Query: %s\n", query)
        fmt.Printf("Limit: %d\n", *limit)
    }
    
    // Create pipeline
    pipe, err := pipeline.New(cfg, registry)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create pipeline: %v\n", err)
        return 1
    }
    defer pipe.Close()
    
    // Execute custom logic
    result, err := executeCustomQuery(ctx, pipe, query, *limit)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        return 1
    }
    
    fmt.Println(result)
    return 0
}

func executeCustomQuery(ctx context.Context, pipe *pipeline.Pipeline, query string, limit int) (string, error) {
    // Your custom logic here
    return fmt.Sprintf("Processed: %s (limit: %d)", query, limit), nil
}
```

### Registering the Subcommand

```go
// In your main package or cmd/subcommand directory
package main

import (
    "context"
    "flag"
    "fmt"
    "os"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
    "your-plugin/subcommands"
)

func main() {
    // Parse command name
    args := os.Args[1:]
    if len(args) == 0 {
        printUsage()
        os.Exit(1)
    }
    
    command := args[0]
    subArgs := args[1:]
    
    // Load configuration
    cfg, err := config.Resolve()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
        os.Exit(1)
    }
    
    ctx := context.Background()
    
    // Dispatch to subcommand
    var exitCode int
    switch command {
    case "custom":
        exitCode = subcommands.RunCustom(ctx, cfg, runtime.DefaultRegistry)
    case "chat":
        // Built-in chat
    case "serve":
        // Built-in serve
    default:
        fmt.Fprintf(os.Stderr, "Unknown command: %s\n", command)
        printUsage()
        exitCode = 1
    }
    
    os.Exit(exitCode)
}
```

## Complete Example: Statistics Subcommand

Here's a complete example of a custom statistics subcommand:

### Project Structure

```
custom-stats/
├── go.mod
├── stats.go
└── README.md
```

### stats.go

```go
package stats

import (
    "context"
    "encoding/json"
    "flag"
    "fmt"
    "os"
    "text/tabwriter"

    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

type StatsOptions struct {
    JSON        bool
    Verbose     bool
    MemoryOnly  bool
    RuntimeOnly bool
}

func RunStats(ctx context.Context, cfg config.Config, registry runtime.Registry, opts StatsOptions) int {
    // Create pipeline
    pipe, err := pipeline.New(cfg, registry)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create pipeline: %v\n", err)
        return 1
    }
    defer pipe.Close()
    
    // Gather statistics
    var stats map[string]interface{}
    
    if !opts.MemoryOnly {
        // Get memory stats
        memoryStats, err := pipe.GetMemoryStats(ctx)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Failed to get memory stats: %v\n", err)
            return 1
        }
        stats = memoryStats
    }
    
    if !opts.RuntimeOnly {
        // Add runtime info
        stats["runtime_backend"] = cfg.Runtime.Backend
        stats["max_tokens"] = cfg.Runtime.Defaults.MaxTokens
        stats["temperature"] = cfg.Runtime.Defaults.Temperature
    }
    
    // Output
    if opts.JSON {
        encoder := json.NewEncoder(os.Stdout)
        encoder.SetIndent("", "  ")
        if err := encoder.Encode(stats); err != nil {
            fmt.Fprintf(os.Stderr, "JSON encode error: %v\n", err)
            return 1
        }
    } else {
        printStatsTable(stats, opts.Verbose)
    }
    
    return 0
}

func printStatsTable(stats map[string]interface{}, verbose bool) {
    w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
    fmt.Fprintln(w, "Metric\tValue\t")
    fmt.Fprintln(w, "------\t-----\t")
    
    for key, value := range stats {
        if !verbose && isInternalMetric(key) {
            continue
        }
        fmt.Fprintf(w, "%s\t%v\t\n", key, value)
    }
    
    w.Flush()
}

func isInternalMetric(key string) bool {
    internal := map[string]bool{
        "vector_enabled": true,
        "rag_enabled":    true,
    }
    return internal[key]
}
```

### Command Registration

```go
package main

import (
    "context"
    "flag"
    "fmt"
    "os"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
    "your-plugin/stats"
)

func main() {
    // Custom subcommand flags
    jsonFlag := flag.Bool("json", false, "Output as JSON")
    verboseFlag := flag.Bool("v", false, "Verbose output")
    flag.Parse()
    
    subArgs := flag.Args()
    
    if len(subArgs) < 1 || subArgs[0] != "stats" {
        // Not our command
        return
    }
    
    // Remaining args after "stats"
    remaining := subArgs[1:]
    
    cfg, _ := config.Resolve()
    ctx := context.Background()
    
    opts := stats.StatsOptions{
        JSON:    *jsonFlag,
        Verbose: *verboseFlag,
    }
    
    exitCode := stats.RunStats(ctx, cfg, runtime.DefaultRegistry, opts)
    os.Exit(exitCode)
}
```

## Flag Parsing Conventions

### Standard Flag Pattern

```go
func RunCommand(ctx context.Context, cfg config.Config, opts CommandOptions) int {
    // Define flags
    flag.StringVar(&opts.Output, "o", "output.txt", "Output file")
    flag.IntVar(&opts.Limit, "limit", 10, "Maximum results")
    flag.BoolVar(&opts.Verbose, "v", false, "Verbose output")
    flag.BoolVar(&opts.Force, "f", false, "Force overwrite")
    
    // Parse
    flag.Parse()
    args := flag.Args()
    
    if len(args) < 1 {
        fmt.Fprintf(os.Stderr, "Usage: OpenEye command [flags] <input>\n")
        flag.Usage()
        return 1
    }
    
    // ... implementation
}
```

### Environment Variable Support

```go
func RunCommand(ctx context.Context, cfg config.Config, opts CommandOptions) int {
    // Apply environment variable overrides
    if outputEnv := os.Getenv("APP_OUTPUT"); outputEnv != "" && opts.Output == "" {
        opts.Output = outputEnv
    }
    
    if limitEnv := os.Getenv("APP_LIMIT"); limitEnv != "" {
        if limit, err := strconv.Atoi(limitEnv); err == nil {
            opts.Limit = limit
        }
    }
    
    // ... implementation
}
```

## Output Formatting

### Structured Output

```go
type Result struct {
    ID      string
    Name    string
    Score   float64
    Created string
}

func printResults(results []Result, format string) {
    switch format {
    case "json":
        encoder := json.NewEncoder(os.Stdout)
        encoder.SetIndent("", "  ")
        encoder.Encode(results)
        
    case "csv":
        fmtName,Score,.Println("ID,Created")
        for _, r := range results {
            fmt.Printf("%s,%s,%.4f,%s\n", r.ID, r.Name, r.Score, r.Created)
        }
        
    default:
        // Table format
        w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
        fmt.Fprintln(w, "ID\tName\tScore\tCreated\t")
        fmt.Fprintln(w, "--\t----\t-----\t-------\t")
        for _, r := range results {
            fmt.Fprintf(w, "%s\t%s\t%.4f\t%s\t\n", r.ID, r.Name, r.Score, r.Created)
        }
        w.Flush()
    }
}
```

### Error Output

```go
func RunCommand(ctx context.Context, cfg config.Config) int {
    err := executeCommand(ctx, cfg)
    if err != nil {
        // Use Stderr for errors
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        
        // Exit with non-zero code
        return 1
    }
    
    return 0
}
```

## Integration with Pipeline

### Using Pipeline in Subcommands

```go
func RunAnalyze(ctx context.Context, cfg config.Config, opts AnalyzeOptions) int {
    pipe, err := pipeline.New(cfg, runtime.DefaultRegistry)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Pipeline error: %v\n", err)
        return 1
    }
    defer pipe.Close()
    
    // Use the pipeline
    result, err := pipe.Respond(ctx, opts.Query, nil, pipeline.Options{
        DisableRAG:  opts.NoRAG,
        DisableSummary: opts.NoSummary,
        RAGLimit:   opts.RAGLimit,
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Generation error: %v\n", err)
        return 1
    }
    
    // Output result
    fmt.Println(result.Text)
    
    // Output stats if verbose
    if opts.Verbose && result.Stats.TokensGenerated > 0 {
        fmt.Fprintf(os.Stderr, "[stats] tokens=%d duration=%v\n",
            result.Stats.TokensGenerated, result.Stats.Duration)
    }
    
    return 0
}
```

## Complete Subcommand Package

Here's a template for a complete subcommand package:

```go
package subcommands

import (
    "context"
    "flag"
    "fmt"
    "os"

    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

const (
    commandName = "custom"
    commandDesc = "A custom subcommand for specialized tasks"
)

var (
    // Flags
    flagVerbose bool
    flagLimit   int
    flagOutput  string
    flagJSON    bool
    
    // Environment variables
    envLimit  = "APP_CUSTOM_LIMIT"
    envOutput = "APP_CUSTOM_OUTPUT"
)

func init() {
    // Register command (if using a command registry)
}

func AddFlags() {
    flag.BoolVar(&flagVerbose, "v", false, "Verbose output")
    flag.IntVar(&flagLimit, "limit", 10, "Limit results")
    flag.StringVar(&flagOutput, "o", "", "Output file")
    flag.BoolVar(&flagJSON, "json", false, "Output as JSON")
}

func RunCommand(ctx context.Context, cfg config.Config, args []string) int {
    // Reset flags
    flagVerbose = false
    flagLimit = 10
    flagOutput = ""
    flagJSON = false
    
    // Parse flags
    flag.CommandLine.Parse(args)
    remainingArgs := flag.CommandLine.Args()
    
    // Apply environment overrides
    if v := os.Getenv(envLimit); v != "" {
        if n, err := fmt.ScanInt(v); err == nil && n > 0 {
            flagLimit = n
        }
    }
    if v := os.Getenv(envOutput); v != "" {
        flagOutput = v
    }
    
    // Validate
    if len(remainingArgs) < 1 {
        fmt.Fprintf(os.Stderr, "Usage: OpenEye %s [flags] <input>\n", commandName)
        return 1
    }
    
    input := remainingArgs[0]
    
    // Execute
    result, err := execute(ctx, cfg, input)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        return 1
    }
    
    // Output
    if flagJSON {
        printJSON(result)
    } else {
        printText(result)
    }
    
    return 0
}

func execute(ctx context.Context, cfg config.Config, input string) (*Result, error) {
    pipe, err := pipeline.New(cfg, runtime.DefaultRegistry)
    if err != nil {
        return nil, err
    }
    defer pipe.Close()
    
    // Your logic here
    return &Result{Input: input}, nil
}

type Result struct {
    Input  string
    Output string
}

func printText(r *Result) {
    fmt.Println(r.Output)
}

func printJSON(r *Result) {
    encoder := json.NewEncoder(os.Stdout)
    encoder.SetIndent("", "  ")
    encoder.Encode(r)
}
```

## Testing Subcommands

### Unit Testing

```go
package subcommands

import (
    "context"
    "testing"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func TestRunCommand(t *testing.T) {
    cfg := config.Default()
    
    // Test normal execution
    exitCode := RunCommand(context.Background(), cfg, []string{})
    if exitCode != 1 {
        t.Errorf("Expected exit code 1 for missing args, got %d", exitCode)
    }
}

func TestCommandWithArgs(t *testing.T) {
    cfg := config.Default()
    
    // Test with arguments
    exitCode := RunCommand(context.Background(), cfg, []string{"test-input"})
    if exitCode != 0 {
        t.Errorf("Expected exit code 0, got %d", exitCode)
    }
}
```

### Integration Testing

```go
func TestCommandIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    cfg, _ := config.Resolve()
    
    // Test with real pipeline
    exitCode := RunCommand(context.Background(), cfg, []string{"--verbose", "test"})
    if exitCode != 0 {
        t.Errorf("Command failed with exit code %d", exitCode)
    }
}
```

## Common Issues

### Flag Parsing Order

```go
// GOOD: Parse flags at the start
func RunCommand(args []string) int {
    flag.Parse()
    
    // Now use the parsed flags
    if *verbose {
        fmt.Println("Debug mode enabled")
    }
    
    return 0
}

// BAD: Use flags before parsing
func RunCommand(args []string) int {
    // NEVER access *verbose here - undefined behavior
    return 0
}
```

### Context Cancellation

```go
func RunCommand(ctx context.Context, args []string) int {
    // Create cancellable context
    ctx, cancel := context.WithCancel(ctx)
    defer cancel()
    
    // Handle interrupts
    go func() {
        <-os.Interrupt
        cancel()
    }()
    
    // Execute with cancellable context
    err := execute(ctx, args)
    if ctx.Err() == context.Canceled {
        fmt.Println("Operation cancelled")
        return 130  // Standard exit code for SIGINT
    }
    
    return 0
}
```

## Checklist for Production

- [ ] Flags defined following Go conventions
- [ ] Environment variable support added
- [ ] Error messages to Stderr
- [ ] Exit codes follow conventions (0=success, 1=error)
- [ ] Context cancellation respected
- [ ] Pipeline properly closed
- [ ] Output formatting options (JSON, table, etc.)
- [ ] Help text provided
- [ ] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Best Practices](best-practices.md)
