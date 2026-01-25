package logging

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"
)

var (
	logFile   *os.File
	logDir    string
	isFileLog bool
)

// Init initializes logging. If toFile is true, logs are written to a file
// in the logs directory instead of stdout. This prevents log output from
// corrupting the TUI.
func Init(toFile bool) error {
	if !toFile {
		// Default: log to stderr (won't interfere with normal stdout)
		log.SetOutput(os.Stderr)
		log.SetFlags(log.Ltime | log.Lshortfile)
		return nil
	}

	// Determine log directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}
	logDir = filepath.Join(homeDir, ".openeye", "logs")

	// Create log directory
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %w", err)
	}

	// Create log file with timestamp
	timestamp := time.Now().Format("2006-01-02")
	logPath := filepath.Join(logDir, fmt.Sprintf("openeye-%s.log", timestamp))

	logFile, err = os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	log.SetOutput(logFile)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	isFileLog = true

	log.Printf("=== OpenEye session started ===")
	return nil
}

// Close closes the log file if one is open.
func Close() {
	if logFile != nil {
		log.Printf("=== OpenEye session ended ===")
		logFile.Close()
		logFile = nil
	}
}

// Discard sets log output to discard all messages.
// Useful for completely silent operation.
func Discard() {
	log.SetOutput(io.Discard)
}

// GetLogDir returns the directory where logs are stored.
func GetLogDir() string {
	return logDir
}

// IsFileLogging returns true if logging is going to a file.
func IsFileLogging() bool {
	return isFileLog
}
