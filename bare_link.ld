OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
  . = 0x80000000;
  _start_text = .;
  .text.init : { *(.text.init) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .uvmif : { *(.uvmif) }
  . = ALIGN(0x1000);
  .text : { *(.text) *(.text.*) }
  . = ALIGN(0x1000);
  .text.startup : { *(.text.startup) }
  . = ALIGN(0x1000);
  _end_text = .;

  . = ALIGN(0x1000);
  .rodata : { *(.rodata*) }

  . = ALIGN(0x1000);
  .model_blob : { KEEP(*(.model_blob)) }
  . = ALIGN(0x1000);
  .tokenizer_blob : { KEEP(*(.tokenizer_blob)) }

  . = ALIGN(0x1000);
  .data : {
    *(.data)
    *(.sdata)
    __global_pointer$ = . + 0x800;
  }

  . = ALIGN(0x1000);
  .bss : {
    *(.bss)
    *(.sbss)
    *(COMMON)
  }

  . = ALIGN(0x1000);
  .tdata : {
    _tdata_begin = .;
    *(.tdata)
    _tdata_end = .;
  }

  .tbss : {
    *(.tbss)
    _tbss_end = .;
  }

  _end = .;
}
