# OpenEye Makefile
# ================
# Wraps llama.cpp build + Go compilation for a single-command experience.
#
#   make setup     — init submodule + build llama.cpp
#   make native    — build OpenEye with native CGo backend
#   make pi-native — build OpenEye with Raspberry Pi 5 optimizations
#   make http      — build OpenEye HTTP-only (no C deps)
#   make clean     — remove build artifacts
#   make test      — run Go tests
#   make bench     — run memory benchmarks
#
# GPU acceleration (optional):
#   make setup CUDA=1      — build llama.cpp with CUDA support
#   make setup VULKAN=1    — build llama.cpp with Vulkan support

NPROC     ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
LLAMA_DIR  = llama.cpp
BUILD_DIR  = $(LLAMA_DIR)/build
PI_NPROC  ?= 4

# GPU flags (set CUDA=1 or VULKAN=1 on the command line)
CMAKE_EXTRA_FLAGS ?=
ifdef CUDA
	CMAKE_EXTRA_FLAGS += -DGGML_CUDA=ON
endif
ifdef VULKAN
	CMAKE_EXTRA_FLAGS += -DGGML_VULKAN=ON
endif

PI_CMAKE_FLAGS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=OFF \
	-DGGML_NATIVE=ON \
	-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
	-DGGML_OPENMP=ON \
	-DGGML_CPU_KLEIDIAI=ON \
	-DLLAMA_BUILD_TOOLS=ON

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all setup llama native pi-llama pi-native http clean clean-llama test bench help

all: native

## setup: Initialize llama.cpp submodule and build static libraries
setup: llama

llama: $(BUILD_DIR)/src/libllama.a

$(BUILD_DIR)/src/libllama.a:
	@if [ ! -f "$(LLAMA_DIR)/CMakeLists.txt" ]; then \
		echo "=> Initializing llama.cpp submodule..."; \
		git submodule update --init --depth 1 llama.cpp; \
	fi
	@echo "=> Building llama.cpp ($(NPROC) threads)..."
	cmake -B $(BUILD_DIR) -S $(LLAMA_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_NATIVE=ON \
		-DGGML_OPENMP=ON \
		-DLLAMA_BUILD_TOOLS=ON \
		$(CMAKE_EXTRA_FLAGS)
	cmake --build $(BUILD_DIR) --config Release -j$(NPROC)
	@echo "=> llama.cpp build complete."
	@echo "   Libraries:"
	@ls -lh $(BUILD_DIR)/src/libllama.a \
		$(BUILD_DIR)/ggml/src/libggml.a \
		$(BUILD_DIR)/tools/mtmd/libmtmd.a 2>/dev/null || true

## native: Build OpenEye with native CGo backend (builds llama.cpp first if needed)
native: llama
	@echo "=> Building OpenEye (native)..."
	CGO_ENABLED=1 go build -tags native -o openeye-native .
	@echo "=> Done: ./openeye-native"

## pi-llama: Rebuild llama.cpp with Raspberry Pi 5 optimizations
pi-llama:
	@if [ ! -f "$(LLAMA_DIR)/CMakeLists.txt" ]; then \
		echo "=> Initializing llama.cpp submodule..."; \
		git submodule update --init --depth 1 llama.cpp; \
	fi
	@echo "=> Rebuilding llama.cpp for Raspberry Pi 5 ($(PI_NPROC) threads)..."
	rm -rf $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -S $(LLAMA_DIR) $(PI_CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) --config Release -j$(PI_NPROC)
	@echo "=> Raspberry Pi 5 llama.cpp build complete."

## pi-native: Build OpenEye with Raspberry Pi 5 optimized native backend
pi-native: pi-llama
	@echo "=> Building OpenEye (native, Raspberry Pi 5 optimized)..."
	CGO_ENABLED=1 go build -tags native -o openeye-native .
	@echo "=> Done: ./openeye-native"

## http: Build OpenEye HTTP-only (no C dependencies)
http:
	@echo "=> Building OpenEye (HTTP backend)..."
	go build -o openeye .
	@echo "=> Done: ./openeye"

## test: Run all Go tests
test:
	go test ./...

## bench: Run memory benchmarks
bench: native
	./openeye-native benchmark

## clean: Remove Go build artifacts
clean:
	rm -f openeye openeye-native OpenEye
	rm -rf benchmark_results/

## clean-llama: Remove llama.cpp build directory
clean-llama:
	rm -rf $(BUILD_DIR)

## clean-all: Remove everything (Go artifacts + llama.cpp build)
clean-all: clean clean-llama

## clean-db: Remove all the DBs
clean-db:
	rm *.duckdb *.db *.duckdb.* *.db-* 

## help: Show available targets
help:
	@echo "OpenEye Build System"
	@echo "===================="
	@echo ""
	@echo "  make setup          Build llama.cpp static libraries"
	@echo "  make native         Build OpenEye with native CGo backend (default)"
	@echo "  make pi-native      Build OpenEye with Raspberry Pi 5 optimizations"
	@echo "  make http           Build OpenEye HTTP-only (no C deps)"
	@echo "  make test           Run Go tests"
	@echo "  make bench          Run memory benchmarks"
	@echo "  make clean          Remove Go build artifacts"
	@echo "  make clean-llama    Remove llama.cpp build directory"
	@echo "  make clean-all      Remove all build artifacts"
	@echo ""
	@echo "Raspberry Pi 5 Build:"
	@echo "  make pi-native      Rebuild llama.cpp with ARM dotprod + fp16 + KleidiAI"
	@echo "  PI_NPROC=$(PI_NPROC) Threads used by the Pi-optimized build"
	@echo ""
	@echo "GPU Options:"
	@echo "  make setup CUDA=1   Build with NVIDIA CUDA support"
	@echo "  make setup VULKAN=1 Build with Vulkan support"
	@echo ""
	@echo "Configuration:"
	@echo "  NPROC=$(NPROC)      Parallel build jobs (auto-detected)"
