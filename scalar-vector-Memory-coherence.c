#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

// --- UART Implementation ---
#define UART_BASE 0x10000000
#define UART_RBR (UART_BASE + 0)
#define UART_THR (UART_BASE + 0)
#define UART_INTERRUPT_ENABLE (UART_BASE + 4)
#define UART_FIFO_CONTROL (UART_BASE + 8)
#define UART_LINE_CONTROL (UART_BASE + 12)
#define UART_MODEM_CONTROL (UART_BASE + 16)
#define UART_LINE_STATUS (UART_BASE + 20)
#define UART_DLAB_LSB (UART_BASE + 0)
#define UART_DLAB_MSB (UART_BASE + 4)

#define CLOCK_FREQUENCY 50000000
#define UART_BITRATE    115200

static void write_reg_u8(uintptr_t addr, uint8_t value) {
    volatile uint8_t *loc_addr = (volatile uint8_t *)addr;
    *loc_addr = value;
}

static uint8_t read_reg_u8(uintptr_t addr) {
    return *(volatile uint8_t *)addr;
}

static int is_transmit_empty() {
    return read_reg_u8(UART_LINE_STATUS) & 0x20;
}

static void write_serial(char a) {
    while (is_transmit_empty() == 0) {};
    write_reg_u8(UART_THR, a);
}

static void init_uart(uint32_t freq, uint32_t baud) {
    uint32_t divisor = freq / (baud << 4);
    write_reg_u8(UART_INTERRUPT_ENABLE, 0x00);
    write_reg_u8(UART_LINE_CONTROL, 0x80);
    write_reg_u8(UART_DLAB_LSB, divisor);
    write_reg_u8(UART_DLAB_MSB, (divisor >> 8) & 0xFF);
    write_reg_u8(UART_LINE_CONTROL, 0x03);
    write_reg_u8(UART_FIFO_CONTROL, 0xC7);
    write_reg_u8(UART_MODEM_CONTROL, 0x20);
}

static void print_uart(const char *str) {
    const char *cur = &str[0];
    while (*cur != '\0') {
        write_serial((uint8_t)*cur);
        ++cur;
    }
}

static void print_uart_int_dec(uint64_t val) {
    char buf[32];
    int i = 0;
    if (val == 0) {
        print_uart("0");
        return;
    }
    while (val > 0) {
        buf[i++] = (val % 10) + '0';
        val /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        write_serial(buf[j]);
    }
}

static void print_uart_float(float val) {
    if (val < 0) {
        print_uart("-");
        val = -val;
    }
    uint64_t int_part = (uint64_t)val;
    float frac_part = val - int_part;
    print_uart_int_dec(int_part);
    print_uart(".");
    uint64_t frac_int = (uint64_t)(frac_part * 1000000);
    if (frac_int == 0) print_uart("000000");
    else print_uart_int_dec(frac_int);
}

// --- Memcpy Implementations ---

void* memcpy_byte(void* dest, const void* src, size_t len) {
  const unsigned char* s = (const unsigned char*)src;
  unsigned char* d = (unsigned char*)dest;
  for (size_t i = 0; i < len; i++) d[i] = s[i];
  return dest;
}

void* memcpy_opt(void* dest, const void* src, size_t len) {
  if ((((uintptr_t)dest | (uintptr_t)src | len) & (sizeof(uintptr_t)-1)) == 0) {
    const uintptr_t* s = (const uintptr_t*)src;
    uintptr_t *d = (uintptr_t*)dest;
    while (d < (uintptr_t*)((char*)dest + len))
      *d++ = *s++;
  } else {
    const char* s = (const char*)src;
    char *d = (char*)dest;
    while (d < (char*)((char*)dest + len))
      *d++ = *s++;
  }
  return dest;
}

// --- Tests ---

void test_coherence() {
    print_uart("\n=============================================\n");
    print_uart("Scalar-Vector Memory Coherence Test\n");
    print_uart("=============================================\n");

    volatile float data[32] __attribute__((aligned(64)));
    for(int i=0; i<32; i++) data[i] = 0.0f;

    float init_val = 100.0f;
    float add_val = 1.0f;
    float expected_val = init_val + add_val;

    // Step 1: Scalar Store
    print_uart("[Step 1] Scalar Store: Writing "); print_uart_float(init_val); print_uart("\n");
    data[0] = init_val;

    asm volatile("fence rw,rw" ::: "memory"); 

    // Step 2: Vector Load
    size_t vl = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v_data = __riscv_vle32_v_f32m1((float*)data, vl);

    // Step 3: Vector Add
    vfloat32m1_t v_add = __riscv_vfmv_v_f_f32m1(add_val, vl);
    v_data = __riscv_vfadd_vv_f32m1(v_data, v_add, vl);

    // Step 4: Vector Store
    print_uart("[Step 4] Vector Store: Writing result "); print_uart_float(expected_val); print_uart("\n");
    __riscv_vse32_v_f32m1((float*)data, v_data, vl);

    asm volatile("fence rw,rw" ::: "memory");

    // Step 5: Scalar Read
    float result = data[0];
    print_uart("[Step 5] Scalar Read:  Read "); print_uart_float(result); print_uart("\n");

    if (result == expected_val) {
        print_uart("\n[SUCCESS] Coherence Check PASSED!\n");
    } else {
        print_uart("\n[FAILURE] Coherence Check FAILED!\n");
        print_uart("Expected: "); print_uart_float(expected_val); print_uart("\n");
        print_uart("Actual:   "); print_uart_float(result); print_uart("\n");
    }
}

void test_memcpy_coherence(const char* name, void* (*memcpy_func)(void*, const void*, size_t)) {
    print_uart("\n---------------------------------------------\n");
    print_uart("Testing "); print_uart(name); print_uart(" Coherence\n");
    print_uart("---------------------------------------------\n");

    #define TEST_SIZE 32
    volatile float src[TEST_SIZE] __attribute__((aligned(64)));
    volatile float dst[TEST_SIZE] __attribute__((aligned(64)));

    // Initialize src with Vector Unit
    size_t vl = __riscv_vsetvl_e32m1(TEST_SIZE);
    vfloat32m1_t v_init = __riscv_vfmv_v_f_f32m1(123.456f, vl);
    __riscv_vse32_v_f32m1((float*)src, v_init, vl);

    print_uart("[Step 1] Vector Store to SRC (123.456)\n");

    // Fence: Vector Store -> Scalar Load (inside memcpy)
    asm volatile("fence rw,rw" ::: "memory");
    
    // Perform Scalar Memcpy
    memcpy_func((void*)dst, (void*)src, TEST_SIZE * sizeof(float));
    print_uart("[Step 2] Scalar Memcpy SRC -> DST\n");

    // Fence: Scalar Store (inside memcpy) -> Vector Load
    asm volatile("fence rw,rw" ::: "memory");

    // Verify with Vector Load from DST
    vfloat32m1_t v_res = __riscv_vle32_v_f32m1((float*)dst, vl);
    float result_arr[TEST_SIZE];
    __riscv_vse32_v_f32m1(result_arr, v_res, vl);
    
    asm volatile("fence rw,rw" ::: "memory");

    int errors = 0;
    for(int i=0; i<TEST_SIZE; i++) {
        // Simple float comparison, exact match expected since we just copied bits
        if(result_arr[i] != 123.456f) {
            errors++;
        }
    }

    if(errors == 0) {
        print_uart("[SUCCESS] "); print_uart(name); print_uart(" passed coherence check.\n");
    } else {
        print_uart("[FAILURE] "); print_uart(name); print_uart(" failed! Found "); print_uart_int_dec(errors); print_uart(" errors.\n");
        print_uart("First mismatch: Expected 123.456, Got "); print_uart_float(result_arr[0]); print_uart("\n");
    }
}

int main() {
    init_uart(CLOCK_FREQUENCY, UART_BITRATE);
    
    test_coherence();
    
    test_memcpy_coherence("Memcpy_Byte", memcpy_byte);
    test_memcpy_coherence("Memcpy_Opt", memcpy_opt);
    
    return 0;
}
