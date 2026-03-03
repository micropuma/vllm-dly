export CUDA_HOME=/home/douliyang/large/env/cu126
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 显式指定编译器变量
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# 告诉 nvcc 使用哪一个 host 编译器（非常关键！）
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-11"
export CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_FLAGS='-allow-unsupported-compiler'"

export MAX_JOBS=8