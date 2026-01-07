.section .tokenizer_blob,"a",@progbits
.balign 64
.global _binary_tokenizer_bin_start
.global _binary_tokenizer_bin_end
_binary_tokenizer_bin_start:
.incbin "tokenizer.bin"
_binary_tokenizer_bin_end:
