# vLLM build from source 

vLLM整体的源码构建过程是比较丝滑的，详细参考[vLLM GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)文档即可。本文档分别测试了两套环境，均成功部署vLLM：   

* RXT5090(BlackWell架构，openbayes租赁)，PyTorch2.8-2204，cuda toolkit 12.6
* RTX3090(Ampere架构，组内服务器)，cuda toolkit 12.0

部署过程中，第一个配置直接参考文档快速配置成功，第二个配置由于cuda toolkit版本较低等问题，需要进行一定修改。具体以第二个配置为例记录vllm源码编译过程。    

## vLLM源码编译过程  
1. 克隆vllm源码
    ```shell
    git clone git@github.com:vllm-project/vllm.git
    ``` 
    我这里使用的是最新版本，v0.16.0  
2. 利用uv创建虚拟环境  
    ```shell
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    ```
    建议参考 [uv 教程](https://realpython.com/python-uv/)。  
3. 服务器的cuda toolkit版本太低，需要自行升级。没有root权限下的升级方案：  
    ```shell
    wget  https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run       
    # 创建一个你自己的安装目录
    mkdir -p $HOME/software/cuda-12.4

    # 执行安装程序 (nvcc要求gcc为12以下，如果是13，需要降级)
    chmod +x ./cuda_12.4.1_550.54.15_linux.run
    ./cuda_12.4.1_550.54.15_linux.run --silent --toolkit --installpath=$HOME/software/cuda-12.4 --no-opengl-libs

    # 环境变量设置  
    export CUDA_HOME=$HOME/software/cuda-12.4
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH    
    ```

    详细细节参考 [非root用户安装CUDA](https://zhuanlan.zhihu.com/p/643760062)。自行选择版本，亲测12.4是ok的。根据vLLM文档，最低要求是12.1。nvcc安装需要对gcc13降级，没有root权限如下操作即可：  
    ```shell
    # 显式指定编译器变量
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

    # 告诉 nvcc 使用哪一个 host 编译器（非常关键！）
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-11"
    export CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_FLAGS='-allow-unsupported-compiler'"
    ```
    然后运行`.run`脚本，亲测有效。  
4. 源码构建vLLM  
    > 注意，vLLM构建默认是512线程，很容易OOM。这里设置export 
    由于想要使用CCACHE缓存，所以必须使用`--no-build-isolation`。需要提前安装好PyTorch(vLLM的pyproject.toml要求是2.10)，如下编译即可：   
    ```shell
    pip install torch==2.10.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
    ```
    然后如下源码编译：   
    ```shell
    CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .
    ```
    安装过程中，出现module not found，无脑uv pip install即可。
5. 耐心等待，跑测试case  
    进/home/douliyang/large/mlsys/vllm-dly/examples/offline_inference/basic目录，运行`python basic.py`。

## References  
1. [vLLM GPU build from source](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
2. [vLLM 编译tricks](https://zhuanlan.zhihu.com/p/1929199538582844390)
3. [vLLM incremental build](https://docs.vllm.ai/en/stable/contributing/incremental_build/#generate-cmakeuserpresetsjson-using-the-helper-script)