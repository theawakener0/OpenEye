package inferbench

import (
	"os"
	"strconv"
	"strings"
)

// readRSS returns the current process RSS (Resident Set Size) in bytes.
// On Linux it reads from /proc/self/status; on other platforms it returns 0.
func readRSS() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			// Format: "VmRSS:    12345 kB"
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseInt(fields[1], 10, 64)
				if err == nil {
					return kb * 1024 // convert kB to bytes
				}
			}
		}
	}
	return 0
}
