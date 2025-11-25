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
RV_MARCH     ?= rv64gcv_zfh_zvfh_zvl512b
RV_MABI      ?= lp64d
SPK          ?= ~/opt/riscv-spike/bin/spike #vlen512 可配置vlen的 spike
PK           ?= pk
MODEL        ?= stories15M.bin
ARGS         ?=

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
# default to the bare-metal harness (override only if you provide your own libc/IO)
RV_BARE_APP    ?= run_bare.c
# bare_syscalls.c
#  $(CURDIR)/bare_link.ld
BARE_MODEL_BIN     ?= $(MODEL)
BARE_TOKENIZER_BIN ?= tokenizer.bin
BARE_BUILD_DIR     ?= build/bare
BARE_MODEL_OBJ      = $(BARE_BUILD_DIR)/model_blob.o
BARE_TOKENIZER_OBJ  = $(BARE_BUILD_DIR)/tokenizer_blob.o

bare_sanitize = $(subst .,_,$(subst /,_,$(1)))
BARE_MODEL_SYM     := $(call bare_sanitize,$(BARE_MODEL_BIN))
BARE_TOKENIZER_SYM := $(call bare_sanitize,$(BARE_TOKENIZER_BIN))
RV_BARE_BIN_DEFS    = -DMODEL_BIN_START=_binary_$(BARE_MODEL_SYM)_start \
	-DMODEL_BIN_END=_binary_$(BARE_MODEL_SYM)_end \
	-DTOKENIZER_BIN_START=_binary_$(BARE_TOKENIZER_SYM)_start \
	-DTOKENIZER_BIN_END=_binary_$(BARE_TOKENIZER_SYM)_end

RV_BARE_INC    ?= /mnt/d/llm/llama2.c-riscv
RV_BARE_CFLAGS = $(RV_CFLAGS_BASE) -O2 -ffreestanding -nostdlib -static -mcmodel=medany \
	-I$(RV_BARE_INC) -Wl,-T,$(RV_BARE_LINKER) -Wl,--gc-sections $(RV_BARE_BIN_DEFS)

# A variant without section GC or any stripping, if Spike can't see tohost/fromhost
RV_BARE_CFLAGS_NOSTRIP = $(RV_CFLAGS_BASE) -O2 -ffreestanding -nostdlib -static -mcmodel=medany \
	-I$(RV_BARE_INC) -Wl,-T,$(RV_BARE_LINKER) $(RV_BARE_BIN_DEFS)

.PHONY: rvbare
rvbare: $(RV_BARE_APP) $(BARE_MODEL_OBJ) $(BARE_TOKENIZER_OBJ) | $(BARE_BUILD_DIR)
	$(RVCC) $(RV_BARE_CFLAGS) $(RV_BARE_CRT) $(RV_BARE_SYSC) $(RV_BARE_APP) \
		$(BARE_MODEL_OBJ) $(BARE_TOKENIZER_OBJ) -lgcc -o run_bare.elf

.PHONY: rvbarenostrip
rvbarenostrip: $(RV_BARE_APP) $(BARE_MODEL_OBJ) $(BARE_TOKENIZER_OBJ) | $(BARE_BUILD_DIR)
	$(RVCC) $(RV_BARE_CFLAGS_NOSTRIP) $(RV_BARE_CRT) $(RV_BARE_SYSC) $(RV_BARE_APP) \
		$(BARE_MODEL_OBJ) $(BARE_TOKENIZER_OBJ) -lgcc -o run_bare.elf

SPK_BARE_FLAGS ?= --isa=$(RV_MARCH)

.PHONY: rvrunbare
rvrunbare: rvbare
	$(SPK) $(SPK_BARE_FLAGS) ./run_bare.elf

.PHONY: rvrunbarenostrip
rvrunbarenostrip: rvbarenostrip
	$(SPK) $(SPK_BARE_FLAGS) ./run_bare.elf

$(BARE_BUILD_DIR):
	mkdir -p $@

$(BARE_MODEL_OBJ): $(BARE_MODEL_BIN) | $(BARE_BUILD_DIR)
	$(RVOBJCOPY) -I binary -O elf64-littleriscv -B riscv \
		--rename-section .data=.model_blob,alloc,load,readonly,data,contents $< $@

$(BARE_TOKENIZER_OBJ): $(BARE_TOKENIZER_BIN) | $(BARE_BUILD_DIR)
	$(RVOBJCOPY) -I binary -O elf64-littleriscv -B riscv \
		--rename-section .data=.tokenizer_blob,alloc,load,readonly,data,contents $< $@

.PHONY: cleanrvbare
cleanrvbare:
	rm -f run_bare.elf
	rm -rf $(BARE_BUILD_DIR)

.PHONY: rvbarenm
rvbarenm:
	$(RISCV_PREFIX)-nm -C run_bare.elf | grep -E 'tohost|fromhost' || true

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
