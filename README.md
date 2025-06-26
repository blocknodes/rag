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

VLLM memory-occupy estimation:

$$ kvcache\ size/token=2*head\_dim*num\_key\_value\_heads*num\_hidden\_layers*(dtype/byte)$$
$$param\ size = param \ size * (dtype/byte) $$
$$total\ mem = kvcache\ size/token * model\ max\ len*concurrency + mode\ size + static\ graph\ etc.$$
