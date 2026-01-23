## 算子测试相关教程

### 软件栈编译器仓库链接：

https://gitlink.org.cn/michaelcjl/llvm-project_riscv

### 安装方法：

```makefile
# 组内编译器安装方法
cd ~/newllvm/build
rm -rf *  # 小心：这会删除 build 目录所有文件

cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;lld;mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
      -DLLVM_OPTIMIZED_TABLEGEN=ON\
      -DCLANG_DEFAULT_CXX_STDLIB=libstdc++ \
      -DCLANG_DEFAULT_RTLIB=compiler-rt \
      -DCLANG_DEFAULT_UNWINDLIB=libunwind \
      -DCLANG_DEFAULT_LINKER=lld \
      ../llvm

ninja
```

### 矩阵乘算子代码：

算子代码内容在matrix_test文件夹下：

- bare_syscalls_uart.c，crt_uart.S为链接文件
- matrix_kernel_1230.h 为RVV，AME的单独算子
- matrix_kernel_noblk_1231.h为组合算子
- test_matrix_kernel_qmatmul_f32.c为测试函数

```cmd
matrix_test$ tree
.
├── Makefile
├── bare_syscalls_uart.c
├── crt_uart.S
├── matrix_kernel_1230.h
├── matrix_kernel_noblk_1231.h
└── test_matrix_kernel_qmatmul_f32.c

1 directory, 6 files
```

**编译算子make指令如下：**

```makefile
make -B rvbareclang RV_BARE_APP=./test_matrix_kernel_qmatmul_f32.c RV_BARE_NAME=./test_matrix_kernel_qmatmul_f32 BARE_PLATFORM=fpga PRINT_WAY=uart BARE_EMBED_BLOBS=0
```

该指令会编译出elf文件和.bin文件，.bin文件用来在FPGA原型上进行测试。会得到类似下图的测试输出：

![image-20260104160645957](C:\Users\18519\AppData\Roaming\Typora\typora-user-images\image-20260104160645957.png)