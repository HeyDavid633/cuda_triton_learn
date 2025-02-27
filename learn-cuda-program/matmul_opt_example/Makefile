NVCC = nvcc
NVCC_FLAGS = -arch=sm_89 -O3
# TARGET = matmul_v5
TARGET = transposeTensor

# all 目标依赖于 $(TARGET)，但不直接定义规则
all: clean $(TARGET) 

# 编译规则：从 $(TARGET).cu 生成 $(TARGET)
# $< 是自动变量，表示 第一个依赖文件，即 hardware_info.cu
# $@ 是自动变量，表示 目标文件，即 hardware_info
$(TARGET): $(TARGET).cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# 伪目标声明
.PHONY: all run clean