// Copyright OpenHW Group contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#define UART_BASE 0x10000000

#define UART_RBR UART_BASE + 0
#define UART_THR UART_BASE + 0
#define UART_INTERRUPT_ENABLE UART_BASE + 4
#define UART_INTERRUPT_IDENT UART_BASE + 8
#define UART_FIFO_CONTROL UART_BASE + 8
#define UART_LINE_CONTROL UART_BASE + 12
#define UART_MODEM_CONTROL UART_BASE + 16
#define UART_LINE_STATUS UART_BASE + 20
#define UART_MODEM_STATUS UART_BASE + 24
#define UART_DLAB_LSB UART_BASE + 0

void write_reg_u8(uintptr_t addr, uint8_t value);
uint8_t read_reg_u8(uintptr_t addr);
int is_transmit_empty(void);
char is_transmit_empty_altera(void);
int is_receive_empty(void);
void write_serial(char a);
int read_serial(uint8_t *res);
void init_uart(uint32_t freq, uint32_t baud);
void print_uart(const char *str);
void bin_to_hex(uint8_t inp, uint8_t res[2]);
extern uint8_t bin_to_hex_table[16];
void print_uart_int(uint32_t val);
void print_uart_addr(uint64_t addr);
void print_uart_byte(uint8_t byte);
// print_uart_hex is in uart_helper.c (should be static there) or here?
// The linker error said duplicate symbol print_uart_hex.
// If it's in uart_helper.c, it's not in uart.c.

#define UART_DLAB_MSB UART_BASE + 4

void init_uart(uint32_t freq, uint32_t baud);

int read_serial(uint8_t *res);

void print_uart(const char* str);

void print_uart_int(uint32_t addr);

void print_uart_addr(uint64_t addr);

void print_uart_byte(uint8_t byte);
