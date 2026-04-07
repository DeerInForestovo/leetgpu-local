NVCC       := nvcc
NVCC_FLAGS := -O2 -std=c++17 -I.
ARCH       := -arch=sm_89    # RTX 4060 (Ada Lovelace)
BUILD_DIR  := build

.PHONY: clean help list

# ---- make problem-vector-add → build & run ----
problem-%: main.cu problems/problem-%.cu common/*.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $(BUILD_DIR)/$@ main.cu problems/$@.cu
	@echo ""
	@./$(BUILD_DIR)/$@

# ---- make build-problem-vector-add → build only ----
build-problem-%: main.cu problems/problem-%.cu common/*.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $(BUILD_DIR)/problem-$* main.cu problems/problem-$*.cu
	@echo "Built: $(BUILD_DIR)/problem-$*"

# ---- make debug-problem-vector-add → build with debug info ----
debug-problem-%: main.cu problems/problem-%.cu common/*.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -G -g -std=c++17 -I. $(ARCH) -o $(BUILD_DIR)/problem-$*_debug main.cu problems/problem-$*.cu
	@echo "Built (debug): $(BUILD_DIR)/problem-$*_debug"

# ---- make profile-problem-vector-add → build & profile with nsys ----
profile-problem-%: main.cu problems/problem-%.cu common/*.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -lineinfo $(ARCH) -o $(BUILD_DIR)/problem-$*_profile main.cu problems/problem-$*.cu
	nsys profile -o $(BUILD_DIR)/problem-$*_report ./$(BUILD_DIR)/problem-$*_profile

clean:
	rm -rf $(BUILD_DIR)

list:
	@echo "Available problems:"
	@ls problems/problem-*.cu 2>/dev/null | sed 's|problems/||;s|\.cu$$||' | while read p; do \
		echo "  make $$p"; \
	done

help:
	@echo "LeetGPU Local - CUDA Practice Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make problem-vector-add          Build and run"
	@echo "  make build-problem-vector-add    Build only"
	@echo "  make debug-problem-vector-add    Build with debug symbols (-G -g)"
	@echo "  make profile-problem-vector-add  Build & profile with nsys"
	@echo "  make list                        List all available problems"
	@echo "  make clean                       Remove build artifacts"
	@echo "  make help                        Show this help"
