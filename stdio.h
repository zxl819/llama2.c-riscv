#ifndef _STDIO_H
#define _STDIO_H

#include <stdarg.h>

// bare_syscalls.c implements a minimal printf-like function if we want, 
// but usually we use uart printing. 
// However, to satisfy the compiler, we declare printf.
// Note: bare_syscalls.c does NOT seem to implement full printf, 
// but it has sprintf/vsprintf.
// We can map printf to a simple wrapper that calls vsprintf then writes to UART/syscall.

int printf(const char* format, ...);
int sprintf(char* str, const char* format, ...);
int vsprintf(char* str, const char* format, va_list ap);

#endif
