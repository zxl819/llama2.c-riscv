.section .model_blob,"a",@progbits
.balign 64
.global _binary_stories15M_bin_start
.global _binary_stories15M_bin_end
_binary_stories15M_bin_start:
.incbin "stories15M.bin"
_binary_stories15M_bin_end:
