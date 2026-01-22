package subcommands

import (
	"fmt"
	"OpenEye/internal/config"
	"gopkg.in/yaml.v3"
)

// RunConfig displays the current configuration.
func RunConfig(cfg config.Config) int {
	fmt.Println("=== OpenEye Configuration ===")
	
	// Convert config to YAML for readable display
	data, err := yaml.Marshal(cfg)
	if err != nil {
		fmt.Printf("Error marshaling config: %v\n", err)
		return 1
	}

	fmt.Println(string(data))
	return 0
}
