OUTPUT_ARCH("riscv")
ENTRY(_start)

/*
修改说明：
- 移除了独立的 SIG 和 PERF 内存区域，只保留一个连续的 ROM 区域。
- 将 .signature 和 .perfdata 段也放入 ROM 区域，使其紧跟在其他数据之后。
- 这样生成的 .bin 文件将是完全紧凑的，其大小约等于 text + data + bss + signature + perfdata 的总和。
*/

MEMORY
{
  /* 只定义一个足够大的连续内存区域 */
  ROM  (rwx) : ORIGIN = 0x80000000, LENGTH = 384M

}

  /* --- 新增部分：定义硬件外设地址 --- */
  UART0_BASE = 0x10000000;

SECTIONS
{
  . = ORIGIN(ROM);

  /* 代码和只读数据段 */
  .text.init : { *(.text.init) } > ROM
  .text      : { *(.text .text.*) } > ROM
  .rodata    : { *(.rodata .rodata.* .srodata .srodata.*) } > ROM
  .rodata.str1.1 : { *(.rodata.str1.1) } > ROM

  /* 显式放置模型和分词器数据，确保它们在 _end 之前 */
  /* NOTE: 输入 .o 里该 section 可能只有 1-byte 对齐；这里强制输出段 64B 对齐，避免 float/vle32 触发未对齐访问陷阱 */
  .model_blob ALIGN(64) : { KEEP(*(.model_blob)) } > ROM
  .tokenizer_blob ALIGN(64) : { KEEP(*(.tokenizer_blob)) } > ROM

  /* 已初始化数据段 */
  . = ALIGN(64);
  .data : { *(.data .data.*) } > ROM

  . = ALIGN(16);
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > ROM

  /* 矩阵段 */
  . = ALIGN(64);
  _matA_start = .;
  .matA : { *(.matA) } > ROM
  _matA_end = .;  ASSERT(SIZEOF(.matA) <= 0x200, "matA too big (>0x200)")

  . = ALIGN(64);
  _matB_start = .;
  .matB : { *(.matB) } > ROM
  _matB_end = .;  ASSERT(SIZEOF(.matB) <= 0x200, "matB too big (>0x200)")

  . = ALIGN(64);
  _matC_start = .;
  .matC : { *(.matC) } > ROM
  _matC_end = .;  ASSERT(SIZEOF(.matC) <= 0x800, "matC too big (>0x800)")
    . = ALIGN(64);
  _matD_start = .;
  .matD : { *(.matD) } > ROM
  _matD_end = .;  ASSERT(SIZEOF(.matD) <= 0x800, "matD too big (>0x800)")

  . = ALIGN(64);
  _matNOLOAD_start = .;
  .matNOLOAD(NOLOAD) : { *(.matNOLOAD) } > ROM
  _matNOLOAD_end = .;  ASSERT(SIZEOF(.matNOLOAD) <= 0x200, "matNOLOAD too big (>0x200)")

  /* 未初始化数据段 */
  . = ALIGN(16);
  .bss (NOLOAD) : { /* BSS段不占用文件大小，标记为NOLOAD */
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
  } > ROM

  .tdata : { _tdata_begin = .; *(.tdata .tdata.*) _tdata_end = .; } > ROM
  .tbss  (NOLOAD) : { *(.tbss .tbss.*) _tbss_end = .; } > ROM

  /* --- 修改点：将 signature 和 perfdata 放入 ROM 区域 --- */
  /* 将它们紧跟在 .bss 段之后 */
  .signature (NOLOAD) : {
    begin_signature = .;
    *(.scdata.output)
    end_signature = .;
  } > ROM

  .perfdata (NOLOAD) : {
    begin_perf_data = .;
    *(.pfdata.output)
    end_perf_data = .;
  } > ROM

  _end = .;

  /* Heap region: from end of image to end of RAM region */
  __heap_start = _end;
  __heap_end = ORIGIN(ROM) + LENGTH(ROM);

  /* 尺寸检查：确保所有内容都在定义的 ROM 区域内 */
  ASSERT(_end - ORIGIN(ROM) <= LENGTH(ROM), "ROM overflow: enlarge LENGTH in MEMORY block")
}