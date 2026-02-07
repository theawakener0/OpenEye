//go:build native

// build_native.go registers the native llama.cpp adapter with the runtime
// registry when building with -tags native.
package native

import "OpenEye/internal/runtime"

func init() {
	runtime.Register("native", newNativeAdapter)
}
