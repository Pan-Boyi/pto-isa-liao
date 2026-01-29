# PTO Runtime2 Makefile
#
# Builds the PTO Runtime2 library with buffer management
# Based on: docs/runtime_buffer_manager_methods.md

CC = gcc
AR = ar
CFLAGS = -Wall -Wextra -O3 -fPIC -std=c11
DEBUG_FLAGS = -g -DDEBUG -O0
INCLUDES = -I. -I./runtime_a2a3_sim/core_model

# Source files
SRCS = \
	pto_shared_memory.c \
	pto_ring_buffer.c \
	pto_tensormap.c \
	pto_scheduler.c \
	pto_orchestrator.c \
	pto_runtime2.c \
	pto_runtime2_sim.c

OBJS = $(SRCS:.c=.o)

# Headers
HEADERS = \
	pto_runtime2_types.h \
	pto_shared_memory.h \
	pto_ring_buffer.h \
	pto_tensormap.h \
	pto_scheduler.h \
	pto_orchestrator.h \
	pto_runtime2.h \
	pto_runtime2_sim.h

# Output library
LIB_NAME = libpto_runtime2
STATIC_LIB = $(LIB_NAME).a
SHARED_LIB = $(LIB_NAME).so

# Core model library (optional)
CORE_MODEL_DIR = runtime_a2a3_sim/core_model
CORE_MODEL_LIB = $(CORE_MODEL_DIR)/liba2a3_core.a

# Test programs
TEST_DIR = tests
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:.c=)

.PHONY: all clean test debug lib shared core_model install help

all: lib

# Build static library
lib: $(STATIC_LIB)

$(STATIC_LIB): $(OBJS)
	$(AR) rcs $@ $^

# Build shared library
shared: $(SHARED_LIB)

$(SHARED_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

# Compile source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Debug build
debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean lib

# Build with core model integration
core_model: CFLAGS += -DA2A3_CORE_SIM_AVAILABLE
core_model: $(CORE_MODEL_LIB)
core_model: lib

$(CORE_MODEL_LIB):
	$(MAKE) -C $(CORE_MODEL_DIR) lib

# Test targets
test: lib
	@mkdir -p $(TEST_DIR)
	@echo "Running tests..."
	@if [ -f $(TEST_DIR)/test_runtime2.c ]; then \
		$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test_runtime2 \
			$(TEST_DIR)/test_runtime2.c -L. -lpto_runtime2 && \
		./$(TEST_DIR)/test_runtime2; \
	else \
		echo "No tests found. Creating example test..."; \
	fi

# Create test directory and example test
$(TEST_DIR)/test_runtime2.c:
	@mkdir -p $(TEST_DIR)
	@echo '#include "pto_runtime2.h"' > $@
	@echo '#include <stdio.h>' >> $@
	@echo '' >> $@
	@echo 'int main(void) {' >> $@
	@echo '    printf("PTO Runtime2 Test\\n");' >> $@
	@echo '    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_SIMULATE);' >> $@
	@echo '    if (!rt) {' >> $@
	@echo '        printf("FAILED: Could not create runtime\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '    printf("Runtime created successfully\\n");' >> $@
	@echo '    pto2_runtime_print_stats(rt);' >> $@
	@echo '    pto2_runtime_destroy(rt);' >> $@
	@echo '    printf("PASSED\\n");' >> $@
	@echo '    return 0;' >> $@
	@echo '}' >> $@

# Clean
clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)
	rm -f $(TEST_DIR)/test_runtime2

clean-all: clean
	$(MAKE) -C $(CORE_MODEL_DIR) clean

# Installation
PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/pto_runtime2
	install -m 644 $(STATIC_LIB) $(PREFIX)/lib/
	install -m 644 $(HEADERS) $(PREFIX)/include/pto_runtime2/

.PHONY: install

# Help
help:
	@echo "PTO Runtime2 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build static library (default)"
	@echo "  lib        - Build static library"
	@echo "  shared     - Build shared library"
	@echo "  debug      - Build with debug symbols"
	@echo "  core_model - Build with A2A3 core model integration"
	@echo "  test       - Build and run tests"
	@echo "  clean      - Remove build artifacts"
	@echo "  clean-all  - Remove all artifacts including core model"
	@echo "  install    - Install to PREFIX (default: /usr/local)"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CC         - C compiler (default: gcc)"
	@echo "  CFLAGS     - Additional compiler flags"
	@echo "  PREFIX     - Installation prefix (default: /usr/local)"
