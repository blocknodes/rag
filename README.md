# rag
```
#!/bin/bash
sh Miniconda3-py310_25.3.1-1-Linux-x86_64.sh 
eval "$(/root/miniconda3/bin/conda shell.YOUR_SHELL_NAME hook)"
cd Qwen-Agent/
pip install -e ./"[gui,rag,code_interpreter,mcp]"  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  /workspace/ragflow_sdk-0.18.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
PYTHONPATH=../Qwen-Agent/:$PYTHONPATH  ./cmd.sh
```