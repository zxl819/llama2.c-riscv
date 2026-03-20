# -----------------------------------------------------------------------------
# Minimal AICPU_test delivery Makefile
#
# Goals:
# 1) Only compile/link files under AICPU_test/
# 2) No dependency on repository-root sources or absolute host paths
# 3) Keep bare-metal flow simple for handoff
# -----------------------------------------------------------------------------

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CODE_DIR := $(abspath $(THIS_DIR)/..)
BUILD_DIR ?= $(THIS_DIR)/build

RISCV_PREFIX ?= riscv64-unknown-elf
RVCC      ?= $(RISCV_PREFIX)-gcc
RVOBJCOPY ?= $(RISCV_PREFIX)-objcopy
RVOBJDUMP ?= $(RISCV_PREFIX)-objdump
# libgcc: prefer a local copy bundled in THIS_DIR for maximum portability.
# Run `make setup` once to copy the system libgcc.a here.
LOCAL_LIBGCC := $(THIS_DIR)/libgcc.a
SYS_LIBGCC   ?= /usr/lib/gcc/riscv64-unknown-elf/13.2.0/libgcc.a

# Toolchain selector: gcc | clang
TOOLCHAIN ?= gcc

# Clang cross settings (compatible with your previous clang flow)
RV_BARE_CLANG ?= ~/newllvm/test_llvm/build/bin/clang
RV_CLANG_TARGET ?= riscv64-unknown-elf
CLANG_GCC_TOOLCHAIN ?=
RV_CLANG_EXTRA_FLAGS ?= -menable-experimental-extensions
RV_CLANG_VLEN_FLAGS ?= \
	-mllvm -riscv-v-vector-bits-min=512 \
	-mllvm -riscv-v-vector-bits-max=512

RV_MARCH ?= rv64gcv_zfh_zvfh_zvl512b
RV_MABI  ?= lp64d

SPK ?= spike

# Select app from AICPU_test/code/*.c (without suffix)
# Available examples:
#   APP=matmul_matrix_inline_16x16
#   APP=test_matrix_kernel_qmatmul_f32
#   APP=test_softmax_rvv
#   APP=softmax
APP ?= matmul_matrix_inline_16x16
APP_SRC := $(firstword \
	$(wildcard $(THIS_DIR)/$(APP).c) \
	$(wildcard $(CODE_DIR)/$(APP).c))

ifeq ($(wildcard $(APP_SRC)),)
$(error APP source not found. Tried: $(THIS_DIR)/$(APP).c and $(CODE_DIR)/$(APP).c)
endif

CRT_SRC      := $(THIS_DIR)/crt_uart.S
SYSCALLS_SRC := $(THIS_DIR)/bare_syscalls_uart.c
LINKER_SCRIPT := $(THIS_DIR)/test_compact.ld

# Base support: UART only (required by all apps)
SUPPORT_OBJS := $(BUILD_DIR)/uart.o

# Matrix/RVV kernel support: only needed by test_matrix_kernel_qmatmul_f32
# Enable with MATRIX_SUPPORT=1, e.g.:
#   make rvbare APP=test_matrix_kernel_qmatmul_f32 MATRIX_SUPPORT=1
MATRIX_SUPPORT ?= 0
ifeq ($(MATRIX_SUPPORT),1)
SUPPORT_OBJS += \
  $(BUILD_DIR)/matrix_kernel_rvv_accum.o \
  $(BUILD_DIR)/matrix_kernel_matmul.o \
  $(BUILD_DIR)/matrix_kernel_noblk.o
endif

ELF  := $(BUILD_DIR)/$(APP).elf
BIN  := $(BUILD_DIR)/$(APP).bin
DUMP := $(BUILD_DIR)/$(APP).dump

COMMON_CFLAGS := -march=$(RV_MARCH) -mabi=$(RV_MABI) \
  -O0 -g -ffreestanding -nostdlib -static -mcmodel=medany \
  -I$(CODE_DIR) -I$(THIS_DIR)

CLANG_TC_FLAG := $(if $(strip $(CLANG_GCC_TOOLCHAIN)),--gcc-toolchain=$(CLANG_GCC_TOOLCHAIN),)

ifeq ($(TOOLCHAIN),clang)
BARE_CC := $(RV_BARE_CLANG) --target=$(RV_CLANG_TARGET) $(CLANG_TC_FLAG)
BARE_CFLAGS := $(COMMON_CFLAGS) $(RV_CLANG_EXTRA_FLAGS) $(RV_CLANG_VLEN_FLAGS)
else
BARE_CC := $(RVCC)
BARE_CFLAGS := $(COMMON_CFLAGS)
endif

LINK_FLAGS := -Wl,-T,$(LINKER_SCRIPT) -Wl,--gc-sections

SPIKE_FLAGS ?= --isa=$(RV_MARCH)

# Use local copy if present, otherwise fall back to system path
ifeq ($(wildcard $(LOCAL_LIBGCC)),$(LOCAL_LIBGCC))
GCC_RUNTIME_LIB := $(LOCAL_LIBGCC)
else
GCC_RUNTIME_LIB := $(SYS_LIBGCC)
endif

.PHONY: all rvbare rvbareclang dump run clean info setup

all: rvbare

setup:
	@echo "Copying libgcc.a from $(SYS_LIBGCC) ..."
	cp $(SYS_LIBGCC) $(LOCAL_LIBGCC)
	@echo "Done: $(LOCAL_LIBGCC)"

info:
	@echo "THIS_DIR=$(THIS_DIR)"
	@echo "CODE_DIR=$(CODE_DIR)"
	@echo "APP=$(APP)"
	@echo "APP_SRC=$(APP_SRC)"
	@echo "TOOLCHAIN=$(TOOLCHAIN)"
	@echo "BARE_CC=$(BARE_CC)"
	@echo "RVCC=$(RVCC)"
	@echo "LOCAL_LIBGCC=$(LOCAL_LIBGCC)"
	@echo "SYS_LIBGCC=$(SYS_LIBGCC)"
	@echo "GCC_RUNTIME_LIB=$(GCC_RUNTIME_LIB)"
	@echo "RV_BARE_CLANG=$(RV_BARE_CLANG)"
	@echo "RV_MARCH=$(RV_MARCH)"
	@echo "RV_MABI=$(RV_MABI)"

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/uart.o: $(THIS_DIR)/uart.c $(THIS_DIR)/uart.h | $(BUILD_DIR)
	$(BARE_CC) $(BARE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_kernel_rvv_accum.o: $(THIS_DIR)/matrix_kernel_rvv_accum.c $(THIS_DIR)/matrix_kernel_1230.h | $(BUILD_DIR)
	$(BARE_CC) $(BARE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_kernel_matmul.o: $(THIS_DIR)/matrix_kernel_matmul.c $(THIS_DIR)/matrix_kernel_1230.h | $(BUILD_DIR)
	$(BARE_CC) $(BARE_CFLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_kernel_noblk.o: $(THIS_DIR)/matrix_kernel_noblk.c $(THIS_DIR)/matrix_kernel_noblk_1231.h | $(BUILD_DIR)
	$(BARE_CC) $(BARE_CFLAGS) -c $< -o $@

rvbare: $(APP_SRC) $(CRT_SRC) $(SYSCALLS_SRC) $(SUPPORT_OBJS) | $(BUILD_DIR)
	$(BARE_CC) $(BARE_CFLAGS) $(LINK_FLAGS) \
	  $(CRT_SRC) $(SYSCALLS_SRC) $(APP_SRC) $(SUPPORT_OBJS) $(GCC_RUNTIME_LIB) -o $(ELF)
	$(RVOBJCOPY) -O binary $(ELF) $(BIN)

rvbareclang:
	$(MAKE) rvbare TOOLCHAIN=clang APP=$(APP) MATRIX_SUPPORT=$(MATRIX_SUPPORT)

dump: rvbare
	$(RVOBJDUMP) -S $(ELF) > $(DUMP)

run: rvbare
	$(SPK) $(SPIKE_FLAGS) $(ELF)

clean:
	rm -rf $(BUILD_DIR)

