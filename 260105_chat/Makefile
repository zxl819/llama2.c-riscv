# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm
	$(CC) -O3 -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq

# =============================================================
# RISC-V cross compilation & Spike simulation targets
# Usage examples:
#   make rv            # build RISC-V binaries (run_rv, runq_rv)
#   make rvdebug       # debug build (-g, no optimization)
#   make rvfast        # Ofast build
#   make rvrun MODEL=stories15M.bin ARGS="-n 3"   # run run_rv under spike pk
#   make rvrunq MODEL=stories15M.bin               # run runq_rv under spike pk
#
# Requirements:
#   1) RISC-V toolchain installed (e.g. riscv64-unknown-elf-gcc)
#   2) Spike simulator (spike) and proxy kernel (pk) in PATH
#      If using a Linux userland toolchain (riscv64-linux-gnu-gcc) you can
#      run spike directly on a statically linked binary (if configured), but
#      most setups use: spike pk <program>
#
# Customization variables (override on command line if needed):
#   RISCV_PREFIX : toolchain prefix (default riscv64-unknown-elf)
#   RV_MARCH     : ISA string (default rv64gc)
#   RV_MABI      : ABI (default lp64d)
#   MODEL        : model file passed as first arg (default stories15M.bin)
#   ARGS         : extra runtime args for run_rv / runq_rv
#   SPK          : spike executable (default spike)
#   PK           : proxy kernel executable (default pk)
#
# Example full run:
#   make rvfast
#   make rvrun MODEL=stories15M.bin ARGS="-n 16 -t 4"
# =============================================================

RISCV_PREFIX ?= riscv64-unknown-elf
RVCC         ?= $(RISCV_PREFIX)-gcc
RVOBJCOPY    ?= $(RISCV_PREFIX)-objcopy
RVOBJDUMP    ?= $(RISCV_PREFIX)-objdump
RV_MARCH     ?= rv64gcv_zfh_zvfh_zvl512b
RV_MABI      ?= lp64d
SPK          ?= ~/riscv-isa-sim/build/spike
PK           ?= pk
MODEL        ?= stories15M.bin
ARGS         ?=
# ~/opt/riscv-spike/bin/spike #vlen512 可配置vlen的 spike

# Clang cross sysroot/toolchain (used by rvbareclang)
RISCV_HOME    ?= /home/xl2025/opt/riscv
RISCV_SYSROOT ?= $(RISCV_HOME)/riscv64-unknown-elf
RISCV_SYS_INC ?= -I$(RISCV_SYSROOT)/riscv64-unknown-elf/include -I$(RISCV_HOME)/include -I$(RISCV_SYSROOT)/include
CLANG_GCC_TOOLCHAIN ?= $(RISCV_HOME)

# Common RISC-V compile flags
RV_CFLAGS_BASE = -march=$(RV_MARCH) -mabi=$(RV_MABI)

.PHONY: rv
rv: run.c
	$(RVCC) -O3 $(RV_CFLAGS_BASE) -o run_rv run.c -lm
	$(RVCC) -O3 $(RV_CFLAGS_BASE) -o runq_rv runq.c -lm

.PHONY: rvfast
rvfast: run.c
	$(RVCC) -Ofast $(RV_CFLAGS_BASE) -o run_rv run.c -lm
	$(RVCC) -Ofast $(RV_CFLAGS_BASE) -o runq_rv runq.c -lm

.PHONY: rvdebug
rvdebug: run.c
	$(RVCC) -g $(RV_CFLAGS_BASE) -o run_rv run.c -lm
	$(RVCC) -g $(RV_CFLAGS_BASE) -o runq_rv runq.c -lm

# Optional OpenMP build for RISC-V (requires toolchain support for -fopenmp)
.PHONY: rvomp
rvomp: run.c
	$(RVCC) -Ofast -fopenmp $(RV_CFLAGS_BASE) -o run_rv run.c -lm
	$(RVCC) -Ofast -fopenmp $(RV_CFLAGS_BASE) -o runq_rv runq.c -lm

# Run under Spike + pk (proxy kernel). MODEL is first argument, ARGS are extras.
.PHONY: rvrun
rvrun: rv
	$(SPK) $(PK) ./run_rv $(MODEL) $(ARGS)

.PHONY: rvrunq
rvrunq: rv
	$(SPK) $(PK) ./runq_rv $(MODEL) $(ARGS)

# Clean RISC-V artifacts
.PHONY: cleanrv
cleanrv:
	rm -f run_rv
	rm -f runq_rv

# =============================================================
# RISC-V bare-metal (no pk) using Spike tohost/fromhost
# This links against riscv-dnn's minimal crt and syscalls, and a tiny bare main
# NOTE: run.c needs host file I/O for model/tokenizer; bare-metal demo just prints.
# This section now embeds the MODEL/TOKENIZER blobs directly into run_bare.elf, so
# running `make rvbare && make rvrunbare` produces a self-contained image.
# Customization knobs:
#   BARE_MODEL_BIN      : model checkpoint to embed (defaults to MODEL variable)
#   BARE_TOKENIZER_BIN  : tokenizer to embed (defaults to tokenizer.bin)
#   SPK_BARE_FLAGS      : extra Spike flags (defaults to --isa=$(RV_MARCH))
# -------------------------------------------------------------
RV_BARE_LINKER ?= /mnt/d/riscv-dnn/include/common_spike/test_compact.ld
RV_BARE_CRT    ?= /mnt/d/riscv-dnn/include/common/crt.S
RV_BARE_SYSC   ?= bare_syscalls.c

# Bare-metal platform selection.
# - spike: HTIF tohost/fromhost (printf/syscalls) via bare_syscalls.c
# - fpga : memory-mapped UART only (no HTIF)
BARE_PLATFORM ?= spike

ifeq ($(BARE_PLATFORM),fpga)
RV_BARE_CRT  = $(CURDIR)/crt_uart.S
RV_BARE_SYSC = bare_syscalls_uart.c
endif
# default to the bare-metal harness (override only if you provide your own libc/IO)
RV_BARE_APP    ?= run_bare.c

# Normalize RV_BARE_APP into a source path that exists.
# NOTE: RV_BARE_APP is often provided on the command line, and GNU make will not
# let makefile assignments overwrite it. So we compute RV_BARE_APP_SRC and use
# that in prerequisites/commands.
RV_BARE_APP_SRC := $(RV_BARE_APP)

# If the user passes RV_BARE_APP without an extension (e.g. ./foo), prefer ./foo.c
# when it exists. This avoids make's implicit rules accidentally building a host
# binary with `gcc`.
ifneq ($(suffix $(RV_BARE_APP_SRC)),.c)
ifneq ($(suffix $(RV_BARE_APP_SRC)),.S)
ifneq ($(suffix $(RV_BARE_APP_SRC)),.s)
ifneq ($(wildcard $(RV_BARE_APP_SRC).c),)
RV_BARE_APP_SRC := $(RV_BARE_APP_SRC).c
endif
endif
endif
endif

ifeq ($(wildcard $(RV_BARE_APP_SRC)),)
$(error RV_BARE_APP '$(RV_BARE_APP)' not found. Pass a source file like RV_BARE_APP=matmul_matrix_inline.c)
endif

RV_BARE_NAME   ?= $(basename $(notdir $(RV_BARE_APP_SRC)))
# bare_syscalls.c
#  $(CURDIR)/bare_link.ld
BARE_MODEL_BIN     ?= $(MODEL)
BARE_TOKENIZER_BIN ?= tokenizer.bin
BARE_BUILD_DIR     ?= build/bare
BARE_MODEL_OBJ      = $(BARE_BUILD_DIR)/model_blob.o
BARE_TOKENIZER_OBJ  = $(BARE_BUILD_DIR)/tokenizer_blob.o

# Extra compile-time defines for bare-metal builds (override on command line).
# Example:
#   make rvbareclang RV_BARE_APP=matmul.c BARE_DEFS="-DRVV_HAS_VSTART_CSR=1 -DRVV_FORCE_CLEAR_VSTART=1"
BARE_DEFS ?=

# Operator/unit tests often don't need embedded model/tokenizer blobs.
# Set BARE_EMBED_BLOBS=0 to skip linking those objects and skip the related -D symbols.
BARE_EMBED_BLOBS ?= 1

ifeq ($(BARE_EMBED_BLOBS),1)
RV_BARE_BLOB_DEFS = $(RV_BARE_BIN_DEFS)
RV_BARE_BLOB_OBJS = $(BARE_MODEL_OBJ) $(BARE_TOKENIZER_OBJ)
else
RV_BARE_BLOB_DEFS =
RV_BARE_BLOB_OBJS =
endif

bare_sanitize = $(subst .,_,$(subst /,_,$(1)))
BARE_MODEL_SYM     := $(call bare_sanitize,$(BARE_MODEL_BIN))
BARE_TOKENIZER_SYM := $(call bare_sanitize,$(BARE_TOKENIZER_BIN))
RV_BARE_BIN_DEFS    = -DMODEL_BIN_START=_binary_$(BARE_MODEL_SYM)_start \
	-DMODEL_BIN_END=_binary_$(BARE_MODEL_SYM)_end \
	-DTOKENIZER_BIN_START=_binary_$(BARE_TOKENIZER_SYM)_start \
	-DTOKENIZER_BIN_END=_binary_$(BARE_TOKENIZER_SYM)_end

RV_BARE_INC    ?= /mnt/d/llm/llama2.c-riscv

# Select bare-metal print backend.
# - PRINT_WAY=printf : route output to printf (Spike tohost/fromhost)
# - PRINT_WAY=uart   : use memory-mapped UART (hardware)
# For convenience you can also use PRINT_WAY=1/0.
PRINT_WAY ?= printf

ifeq ($(PRINT_WAY),printf)
PLAT_PRINT_DEF = -DPLAT_PRINT_USE_PRINTF=1
else ifeq ($(PRINT_WAY),1)
PLAT_PRINT_DEF = -DPLAT_PRINT_USE_PRINTF=1
else ifeq ($(PRINT_WAY),uart)
PLAT_PRINT_DEF = -DPLAT_PRINT_USE_PRINTF=0
else ifeq ($(PRINT_WAY),0)
PLAT_PRINT_DEF = -DPLAT_PRINT_USE_PRINTF=0
else
$(error Unknown PRINT_WAY '$(PRINT_WAY)'. Use PRINT_WAY=printf|uart (or 1|0))
endif

RV_BARE_CFLAGS = $(RV_CFLAGS_BASE) -O0 -ffreestanding -nostdlib -static -mcmodel=medany \
	$(PLAT_PRINT_DEF) \
	$(BARE_DEFS) \
	-I$(RV_BARE_INC) -Wl,-T,$(RV_BARE_LINKER) -Wl,--gc-sections $(RV_BARE_BLOB_DEFS)

# A variant without section GC or any stripping, if Spike can't see tohost/fromhost
RV_BARE_CFLAGS_NOSTRIP = $(RV_CFLAGS_BASE) -O0 -ffreestanding -nostdlib -static -mcmodel=medany \
	$(PLAT_PRINT_DEF) \
	$(BARE_DEFS) \
	-I$(RV_BARE_INC) -Wl,-T,$(RV_BARE_LINKER) $(RV_BARE_BLOB_DEFS)

# =============================================================
# Mix-and-link: build selected Matrix sources with clang(Matrix)
#
# Use-case:
#   - RVV/model code is built with one toolchain (e.g. gcc / newer clang)
#   - Matrix ops are built with clang14-based Matrix toolchain
#   - Final link happens here by adding the Matrix .o files to the link line
#
# How to use:
#   make rvbare RV_BARE_APP=run_bare_hw_vector.c \
#        MATRIX_SRCS="/mnt/d/riscv-dnn/src/your_matrix_op.c"
#
# You can also override MATRIX_CC/MATRIX_MARCH on the command line.

# CLANG_HOME 设定后 CC / CXX 指向自编 LLVM。
MATRIX_CLANG_HOME ?= $(HOME)/opt/riscv/riscv-matrix-project/llvm-project
MATRIX_CC ?= $(MATRIX_CLANG_HOME)/build-ninja/bin/clang
MATRIX_CXX ?= $(MATRIX_CLANG_HOME)/build-ninja/bin/clang++
MATRIX_MARCH ?= rv64gcv0p10_zfh0p1
MATRIX_MABI ?= $(RV_MABI)

# riscv-dnn headers (riscv_matrix.h, matrix/matrix_intrinsic.h, etc.)
RVDNN_INC ?= /mnt/d/riscv-dnn/include

# Extra macros for Matrix-only compilation units.
MATRIX_DEFS ?=

# Space-separated list of Matrix sources to compile (absolute or relative paths).
MATRIX_SRCS ?=

MATRIX_BUILD_DIR ?= $(BARE_BUILD_DIR)/matrix

# Extra objects/libs to link into the bare-metal binary.
RV_BARE_EXTRA_OBJS ?=
RV_BARE_EXTRA_LIBS ?=

matrix_obj = $(MATRIX_BUILD_DIR)/$(call bare_sanitize,$(1)).o
MATRIX_OBJS := $(foreach s,$(MATRIX_SRCS),$(call matrix_obj,$(s)))

MATRIX_CFLAGS ?= \
	--target=riscv64-unknown-elf -march=$(MATRIX_MARCH) -mabi=$(MATRIX_MABI) \
	-menable-experimental-extensions -mcmodel=medany \
	-ffreestanding -fno-builtin -nostdlib -static -O3 -g \
	$(MATRIX_DEFS) -I$(RV_BARE_INC) -I$(RVDNN_INC)

ifneq ($(strip $(MATRIX_SRCS)),)
RV_BARE_EXTRA_OBJS += $(MATRIX_OBJS)
endif

$(MATRIX_BUILD_DIR):
	mkdir -p $@

define matrix_compile_rule
$(call matrix_obj,$(1)): $(1) | $(MATRIX_BUILD_DIR)
	$(MATRIX_CC) $(MATRIX_CFLAGS) -c $$< -o $$@
endef

$(foreach s,$(MATRIX_SRCS),$(eval $(call matrix_compile_rule,$(s))))

.PHONY: matrixobjs
matrixobjs: $(MATRIX_OBJS)

.PHONY: rvbare
rvbare: $(RV_BARE_APP_SRC) $(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) | $(BARE_BUILD_DIR)
	$(RVCC) $(RV_BARE_CFLAGS) $(RV_BARE_CRT) $(RV_BARE_SYSC) $(RV_BARE_APP_SRC) \
		$(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) $(RV_BARE_EXTRA_LIBS) -lgcc -o $(RV_BARE_NAME).elf
	$(RVOBJCOPY) -O binary $(RV_BARE_NAME).elf $(RV_BARE_NAME).bin

.PHONY: rvbarenostrip
rvbarenostrip: $(RV_BARE_APP_SRC) $(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) | $(BARE_BUILD_DIR)
	$(RVCC) $(RV_BARE_CFLAGS_NOSTRIP) $(RV_BARE_CRT) $(RV_BARE_SYSC) $(RV_BARE_APP_SRC) \
		$(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) $(RV_BARE_EXTRA_LIBS) -lgcc -o $(RV_BARE_NAME).elf
	$(RVOBJCOPY) -O binary $(RV_BARE_NAME).elf $(RV_BARE_NAME).bin

.PHONY: rvbarechat

rvbarechat: $(RV_BARE_BLOB_OBJS) | $(BARE_BUILD_DIR)
	$(RVCC) $(RV_BARE_CFLAGS) $(RV_BARE_CRT) $(RV_BARE_SYSC) run_bare_chat.c \
		$(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) $(RV_BARE_EXTRA_LIBS) -lgcc -o run_bare_chat.elf
	$(RVOBJCOPY) -O binary run_bare_chat.elf run_bare_chat.bin

SPK_BARE_FLAGS ?= --isa=$(RV_MARCH)

# Optional Spike trace/signature collection for bare-metal runs.
# Defaults match the desired invocation style:
#   spike --isa=... -l --log-commits +signature=build/bare/spike.sig +signature-granularity=32 <elf> > build/bare/spike.log 2>&1
SPK_BARE_LOG_ARGS ?= 
SPK_BARE_SIG_ARGS ?= +signature=$(BARE_BUILD_DIR)/spike.sig +signature-granularity=32
SPK_BARE_POST     ?= > $(BARE_BUILD_DIR)/spike.log 2>&1

# Combine all spike args for bare runs (override any of these vars on command line if needed).
SPK_BARE_ALL_ARGS ?= $(SPK_BARE_FLAGS) $(SPK_BARE_LOG_ARGS) $(SPK_BARE_SIG_ARGS)

#-l --log-commits
.PHONY: rvrunbare
rvrunbare: rvbare
	@mkdir -p $(BARE_BUILD_DIR)
	$(SPK) $(SPK_BARE_ALL_ARGS) ./$(RV_BARE_NAME).elf $(SPK_BARE_POST)

.PHONY: rvrunbarechat
rvrunbarechat: rvbarechat
	@mkdir -p $(BARE_BUILD_DIR)
	$(SPK) $(SPK_BARE_ALL_ARGS) ./run_bare_chat.elf $(SPK_BARE_POST)

.PHONY: rvrunbarenostrip
rvrunbarenostrip: rvbarenostrip
	@mkdir -p $(BARE_BUILD_DIR)
	$(SPK) $(SPK_BARE_ALL_ARGS) ./$(RV_BARE_NAME).elf $(SPK_BARE_POST)

$(BARE_BUILD_DIR):
	mkdir -p $@

BARE_BLOB_CFLAGS ?= $(RV_CFLAGS_BASE) -ffreestanding -nostdlib -static -mcmodel=medany

$(BARE_BUILD_DIR)/model_blob.S: $(BARE_MODEL_BIN) | $(BARE_BUILD_DIR)
	@printf '%s\n' \
		'.section .model_blob,"a",@progbits' \
		'.balign 64' \
		'.global _binary_$(BARE_MODEL_SYM)_start' \
		'.global _binary_$(BARE_MODEL_SYM)_end' \
		'_binary_$(BARE_MODEL_SYM)_start:' \
		'.incbin "$(BARE_MODEL_BIN)"' \
		'_binary_$(BARE_MODEL_SYM)_end:' \
		> $@

$(BARE_BUILD_DIR)/tokenizer_blob.S: $(BARE_TOKENIZER_BIN) | $(BARE_BUILD_DIR)
	@printf '%s\n' \
		'.section .tokenizer_blob,"a",@progbits' \
		'.balign 64' \
		'.global _binary_$(BARE_TOKENIZER_SYM)_start' \
		'.global _binary_$(BARE_TOKENIZER_SYM)_end' \
		'_binary_$(BARE_TOKENIZER_SYM)_start:' \
		'.incbin "$(BARE_TOKENIZER_BIN)"' \
		'_binary_$(BARE_TOKENIZER_SYM)_end:' \
		> $@

$(BARE_MODEL_OBJ): $(BARE_BUILD_DIR)/model_blob.S | $(BARE_BUILD_DIR)
	$(RVCC) $(BARE_BLOB_CFLAGS) -c $< -o $@

$(BARE_TOKENIZER_OBJ): $(BARE_BUILD_DIR)/tokenizer_blob.S | $(BARE_BUILD_DIR)
	$(RVCC) $(BARE_BLOB_CFLAGS) -c $< -o $@

.PHONY: cleanrvbare
cleanrvbare:
	rm -f $(RV_BARE_NAME).elf $(RV_BARE_NAME).bin $(RV_BARE_NAME).dump
	rm -f run_bare_chat.elf run_bare_chat.bin run_bare_chat.dump
	rm -rf $(BARE_BUILD_DIR)

.PHONY: rvbarenm
rvbarenm:
	$(RISCV_PREFIX)-nm -C $(RV_BARE_NAME).elf | grep -E 'tohost|fromhost' || true


# Bare-metal linking using clang as the driver (kept as a convenience target).
RV_BARE_CLANG ?= ~/newllvm/test_llvm/build/bin/clang
RV_BARE_CLANG_MARCH ?= $(RV_MARCH)
RV_BARE_CLANG_O ?= -O1
# Extra flags for the clang driver (e.g. -menable-experimental-extensions).
RV_BARE_CLANG_EXTRA_FLAGS ?=
#clang

# Only meaningful when the selected -march includes V.
RV_BARE_CLANG_VLEN_FLAGS ?= \
	-mllvm -riscv-v-vector-bits-min=512 \
	-mllvm -riscv-v-vector-bits-max=512
# ifeq ($(findstring v,$(RV_BARE_CLANG_MARCH)),)
# RV_BARE_CLANG_VLEN_FLAGS :=
# endif
./matrix_kernel_rvv_accum.o: matrix_kernel_rvv_accum.c
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf -march=$(RV_BARE_CLANG_MARCH) -mabi=$(RV_MABI) -O1 -ffreestanding -nostdlib -static -mcmodel=medany $(RV_BARE_CLANG_VLEN_FLAGS) -I$(RV_BARE_INC) -I. -c $< -o $@

./matrix_kernel_matmul.o: matrix_kernel_matmul.c
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf -march=$(RV_BARE_CLANG_MARCH) -mabi=$(RV_MABI) -O1 -g -mcmodel=medany -I$(RV_BARE_INC) -I/mnt/d/riscv-dnn/include -I/mnt/d/riscv-dnn/test/ops/matmul_llama -DRVDNN_FORCE_STUB_MATRIX=1 -c $< -o $@

./uart.o: ../uart.c
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf -march=$(RV_BARE_CLANG_MARCH) -mabi=$(RV_MABI) -O1 -ffreestanding -nostdlib -static -mcmodel=medany -menable-experimental-extensions -I$(RV_BARE_INC) -c $< -o $@

./matrix_kernel_noblk.o: matrix_kernel_noblk.c
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf -march=$(RV_BARE_CLANG_MARCH) -mabi=$(RV_MABI) -O1 -ffreestanding -nostdlib -static -mcmodel=medany -menable-experimental-extensions -I$(RV_BARE_INC) -c $< -o $@

./matrix_kernel_noblk_CT.o: matrix_kernel_noblk_CT.c
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf -march=$(RV_BARE_CLANG_MARCH) -mabi=$(RV_MABI) -O1 -ffreestanding -nostdlib -static -mcmodel=medany -menable-experimental-extensions -I$(RV_BARE_INC) -c $< -o $@

RV_BARE_EXTRA_OBJS += ./matrix_kernel_rvv_accum.o ./matrix_kernel_matmul.o ./uart.o ./matrix_kernel_noblk.o ./matrix_kernel_noblk_CT.o

rvbareclang: $(RV_BARE_APP_SRC) $(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) | $(BARE_BUILD_DIR)
	$(RV_BARE_CLANG) --target=riscv64-unknown-elf \
	-march=$(RV_BARE_CLANG_MARCH) \
	-mabi=$(RV_MABI) \
	$(RV_BARE_CLANG_O) $(RV_BARE_CLANG_EXTRA_FLAGS) -ffreestanding -nostdlib -static -mcmodel=medany \
	$(PLAT_PRINT_DEF) \
	$(BARE_DEFS) \
	$(RV_BARE_CLANG_VLEN_FLAGS) \
	-I$(RV_BARE_INC) \
	-Wl,-T,$(RV_BARE_LINKER) \
	--gcc-toolchain=$(CLANG_GCC_TOOLCHAIN) \
	$(RV_BARE_BLOB_DEFS) \
	$(RV_BARE_CRT) $(RV_BARE_SYSC) $(RV_BARE_APP_SRC) \
	$(RV_BARE_BLOB_OBJS) $(RV_BARE_EXTRA_OBJS) $(RV_BARE_EXTRA_LIBS) \
	-lgcc -o $(RV_BARE_NAME).elf
	$(RVOBJCOPY) -O binary $(RV_BARE_NAME).elf $(RV_BARE_NAME).bin
rvrunbareclang: rvbareclang
	$(SPK) $(SPK_BARE_ALL_ARGS) ./$(RV_BARE_NAME).elf $(SPK_BARE_POST)

# dump file generation
dump: $(target_dump)
target_dump = $(RV_BARE_NAME).dump
$(target_dump): ./$(RV_BARE_NAME).elf
	$(RVOBJDUMP) -S ./$(RV_BARE_NAME).elf > $(target_dump)
# Convenience target to show current configuration
.PHONY: rvinfo
rvinfo:
	@echo "RISCV_PREFIX=$(RISCV_PREFIX)"
	@echo "RVCC=$(RVCC)"
	@echo "RV_MARCH=$(RV_MARCH)"
	@echo "RV_MABI=$(RV_MABI)"
	@echo "MODEL=$(MODEL)"
	@echo "ARGS=$(ARGS)"
	@which $(RVCC) || echo "WARNING: $(RVCC) not found in PATH" >&2
	@which $(SPK)  || echo "WARNING: spike not found in PATH" >&2
	@which $(PK)   || echo "WARNING: pk not found in PATH" >&2
