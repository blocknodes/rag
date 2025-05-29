from qwen_agent.llm import get_chat_model
import sys
if __name__ == "__main__":
    import argparse
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        'model': 'deepseek-r1:70b',
        'model_server': 'http://10.24.9.6:11434/v1',
        'generate_cfg': {

        },
    })

    content='## train\n- 延续Qwen-VL，Qwen2-VL也采用了3-stage的训练过程：ViT训练 -> 全参数训练 -> LLM指令微调\n- 多样性的数据包括：图像文本对、OCR 数据、图像文本的文章、VQA 数据、视频对话以及图像知识。来源于网站、开源数据集以及人造数据\n- Qwen2-VL的LLM组件使用来自Qwen2的参数进行初始化；而 Qwen2-VL的视觉编码器使用来自DFN的ViT进行初始化，原始ViT中的固定位置嵌入被替换为RoPE-2D\n## Deploy\n- [deploy.md](./source_code/deploy.md)\n'
    ans='CLIP的文本编码器采用GPT-2风格的Transformer编码器，而图像编码器可以选择ResNet或Vision Transformer（ViT）架构。'
    content=f'{content}以上内容能否推出"{ans}"的结论，只需回答yes或者no'
    print(content)

    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'system', 'content': "You are a helpful assistant."},
                {'role': 'user', 'content': content}]

    print(f'{content}\n\n')

    response = llm.chat(messages,stream=False,)
    print(response[0]['content'])