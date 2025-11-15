package subcommands

import (
	"flag"
	"fmt"
	"os"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/context/memory"
)

// RunMemory prints recent conversation turns from persistent storage.
func RunMemory(cfg config.Config, args []string) int {
	fs := flag.NewFlagSet("memory", flag.ContinueOnError)
	limit := fs.Int("n", cfg.Memory.TurnsToUse, "Number of turns to display")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	store, err := memory.NewStore(cfg.Memory.Path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open memory store: %v\n", err)
		return 1
	}
	defer store.Close()

	if *limit <= 0 {
		*limit = cfg.Memory.TurnsToUse
		if *limit <= 0 {
			*limit = 10
		}
	}

	entries, err := store.Recent(*limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read memory: %v\n", err)
		return 1
	}

	if len(entries) == 0 {
		fmt.Println("no memory entries yet")
		return 0
	}

	fmt.Printf("last %d turns:\n", len(entries))
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		fmt.Printf("[%s] %s: %s\n", entry.CreatedAt.Format(time.RFC3339), entry.Role, entry.Content)
	}
	return 0
}
