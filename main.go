package main

import (
	"os"

	"OpenEye/internal/cli"
)

func main() {
	code := cli.Execute()
	os.Exit(code)
}
