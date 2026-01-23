// Minimalized syscalls and tiny libc for Spike tohost, derived from riscv-dnn include/common/syscalls.c
// Removed dependencies on system headers (stdio.h, string.h, limits.h, sys/signal.h)

#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include "/mnt/d/riscv-dnn/include/common/util.h"

#ifndef SIGABRT
#define SIGABRT 6
#endif

#define SYS_write 64
#define SYS_read 63

#undef strcmp

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

register void *thread_pointer asm("tp");

static uintptr_t syscall(uintptr_t which, uintptr_t arg0, uintptr_t arg1, uintptr_t arg2)
{
  volatile uintptr_t magic_mem[8] __attribute__((aligned(64)));
  magic_mem[0] = which;
  magic_mem[1] = arg0;
  magic_mem[2] = arg1;
  magic_mem[3] = arg2;
#ifdef __riscv_atomic
  __sync_synchronize();
#endif
  tohost = (((uint64_t) ((unsigned long int) magic_mem)) << 16) >> 16;
  while (fromhost == 0)
    ;
  fromhost = 0;
#ifdef __riscv_atomic
  __sync_synchronize();
#endif
  return magic_mem[0];
}

#define NUM_COUNTERS 2
static uintptr_t counters[NUM_COUNTERS];
static char* counter_names[NUM_COUNTERS];

void setStats(int enable)
{
  int i = 0;
#define READ_CTR(name) do { \
    while (i >= NUM_COUNTERS) ; \
    uintptr_t csr = read_csr(name); \
    if (!enable) { csr -= counters[i]; counter_names[i] = #name; } \
    counters[i++] = csr; \
  } while (0)

  READ_CTR(mcycle);
  READ_CTR(minstret);

#undef READ_CTR
}

uintptr_t getStats(int counterid)
{
  return counters[counterid];
}

void __attribute__((noreturn)) tohost_exit(uintptr_t code)
{
  tohost = ((((uint64_t) code) << 17) >> 16) | 1;
  __asm__("nop\n\t");
  while (1);
}

uintptr_t __attribute__((weak)) handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32])
{
  tohost_exit(1337);
}

void exit(int code)
{
  tohost_exit(code);
}

void abort()
{
  exit(128 + SIGABRT);
}

static inline size_t my_strlen(const char *s)
{
  const char *p = s;
  while (*p) p++;
  return (size_t)(p - s);
}

static inline size_t my_strnlen(const char *s, size_t n)
{
  const char *p = s;
  while (n-- && *p) p++;
  return (size_t)(p - s);
}

void printstr(const char* s)
{
#if !NOPRINT
  syscall(SYS_write, 1, (uintptr_t)s, my_strlen(s));
#endif
}

int putchar(int ch);

int puts(const char *s)
{
  const char *p = s;
  while (*p) putchar(*p++);
  putchar('\n');
  return 0;
}

#undef putchar
int putchar(int ch)
{
#if !NOPRINT
  static __thread char buf[64] __attribute__((aligned(64)));
  static __thread int buflen = 0;
  buf[buflen++] = (char)ch;
  if (ch == '\n' || buflen == (int)sizeof(buf))
  {
    syscall(SYS_write, 1, (uintptr_t)buf, (uintptr_t)buflen);
    buflen = 0;
  }
#endif
  return 0;
}

void printhex(uint64_t x)
{
  char str[17];
  int i;
  for (i = 0; i < 16; i++)
  {
    str[15-i] = (x & 0xF) + ((x & 0xF) < 10 ? '0' : 'a'-10);
    x >>= 4;
  }
  str[16] = 0;
  printstr(str);
}

static inline void printnum(void (*putch)(int, void**), void **putdat,
                    unsigned long long num, unsigned base, int width, int padc)
{
  unsigned digs[sizeof(num)*8];
  int pos = 0;
  while (1)
  {
    digs[pos++] = num % base;
    if (num < base)
      break;
    num /= base;
  }
  while (width-- > pos)
    putch(padc, putdat);
  while (pos-- > 0)
    putch(digs[pos] + (digs[pos] >= 10 ? 'a' - 10 : '0'), putdat);
}

static unsigned long long getuint(va_list *ap, int lflag)
{
  if (lflag >= 2)
    return va_arg(*ap, unsigned long long);
  else if (lflag)
    return va_arg(*ap, unsigned long);
  else
    return va_arg(*ap, unsigned int);
}

static long long getint(va_list *ap, int lflag)
{
  if (lflag >= 2)
    return va_arg(*ap, long long);
  else if (lflag)
    return va_arg(*ap, long);
  else
    return va_arg(*ap, int);
}

static void vprintfmt(void (*putch)(int, void**), void **putdat, const char *fmt, va_list ap)
{
  register const char* p;
  const char* last_fmt;
  register int ch;
  unsigned long long num;
  int base, lflag, width, precision, altflag;
  char padc;

  while (1) {
    while ((ch = *(const unsigned char *) fmt) != '%') {
      if (ch == '\0')
        return;
      fmt++;
      putch(ch, putdat);
    }
    fmt++;

    last_fmt = fmt;
    padc = ' ';
    width = -1;
    precision = -1;
    lflag = 0;
    altflag = 0;
  reswitch:
    switch (ch = *(const unsigned char *) fmt++) {
    case '-': padc = '-'; goto reswitch;
    case '0': padc = '0'; goto reswitch;
    case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
      for (precision = 0; ; ++fmt) {
        precision = precision * 10 + ch - '0';
        ch = *fmt;
        if (ch < '0' || ch > '9') break;
      }
      goto process_precision;
    case '*': precision = va_arg(ap, int); goto process_precision;
    case '.': if (width < 0) width = 0; goto reswitch;
    case '#': altflag = 1; goto reswitch;
    process_precision:
      if (width < 0) width = precision, precision = -1; goto reswitch;
    case 'l': lflag++; goto reswitch;
    case 'c': putch(va_arg(ap, int), putdat); break;
    case 's':
      if ((p = va_arg(ap, char *)) == NULL) p = "(null)";
      if (width > 0 && padc != '-')
        for (width -= (int)my_strnlen(p, (size_t)precision); width > 0; width--) putch(padc, putdat);
      for (; (ch = *p) != '\0' && (precision < 0 || --precision >= 0); width--) { putch(ch, putdat); p++; }
      for (; width > 0; width--) putch(' ', putdat);
      break;
    case 'd':
      num = (unsigned long long)getint(&ap, lflag);
      if ((long long) num < 0) { putch('-', putdat); num = -(long long) num; }
      base = 10; goto signed_number;
    case 'u': base = 10; goto unsigned_number;
    case 'o': base = 8; goto unsigned_number;
    case 'p': lflag = 1; putch('0', putdat); putch('x', putdat);
    case 'x': base = 16;
    unsigned_number:
      num = getuint(&ap, lflag);
    signed_number:
      printnum(putch, putdat, num, base, width, padc);
      break;
    case '%': putch(ch, putdat); break;
    default:
      putch('%', putdat); fmt = last_fmt; break;
    }
  }
}

static void putch_adapter(int ch, void **unused)
{ (void)unused; putchar(ch); }

int printf(const char* fmt, ...)
{
  va_list ap; va_start(ap, fmt);
  vprintfmt(putch_adapter, 0, fmt, ap);
  va_end(ap);
  return 0;
}
// 将 putbuf 定义为静态辅助函数，放在 sprintf 的外部
static void putbuf(int ch, void **pp) {
    char **p = (char**)pp;
    **p = (char)ch;
    (*p)++;
}
int sprintf(char* str, const char* fmt, ...)
{
  va_list ap;
  char* str0 = str;
  va_start(ap, fmt);

  // Reuse vprintfmt by writing into buffer
  vprintfmt((void*)putbuf, (void**)&str, fmt, ap);

  *str = 0; // Null-terminate the string
  va_end(ap);
  return (int)(str - str0);
}
// int sprintf(char* str, const char* fmt, ...)
// {
//   va_list ap; char* str0 = str; va_start(ap, fmt);
//   // very small sprintf used only for counters print in _init(); keep minimal
//   // Reuse vprintfmt by writing into buffer
//   void putbuf(int ch, void **pp) { char **p = (char**)pp; **p = (char)ch; (*p)++; }
//   vprintfmt((void*)putbuf, (void**)&str, fmt, ap);
//   *str = 0; va_end(ap);
//   return (int)(str - str0);
// }

void* memcpy(void* dest, const void* src, size_t len)
{
  const unsigned char* s = (const unsigned char*)src;
  unsigned char* d = (unsigned char*)dest;
  for (size_t i = 0; i < len; i++) d[i] = s[i];
  return dest;
}

void* memset(void* dest, int byte, size_t len)
{
  unsigned char* d = (unsigned char*)dest;
  for (size_t i = 0; i < len; i++) d[i] = (unsigned char)byte;
  return dest;
}

size_t strlen(const char *s)
{ return my_strlen(s); }

size_t strnlen(const char *s, size_t n)
{ return my_strnlen(s, n); }

int strcmp(const char* s1, const char* s2)
{
  unsigned char c1, c2;
  do { c1 = *s1++; c2 = *s2++; } while (c1 != 0 && c1 == c2);
  return c1 - c2;
}

char* strcpy(char* dest, const char* src)
{
  char* d = dest; while ((*d++ = *src++)) ; return dest;
}

long atol(const char* str)
{
  long res = 0; int sign = 0;
  while (*str == ' ') str++;
  if (*str == '-' || *str == '+') { sign = *str == '-'; str++; }
  while (*str) { res *= 10; res += *str++ - '0'; }
  return sign ? -res : res;
}

int getchar()
{
  char buf[1];
  long n = syscall(SYS_read, 0, (uintptr_t)buf, 1);
  if (n > 0) return buf[0];
  return -1;
}
