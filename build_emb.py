# Reference: https://platform.openai.com/docs/guides/function-calling
import json
import os
import sys
from qwen_agent.llm import get_chat_model

from qwen_agent.llm import get_chat_model
def read_markdown_file(file_path):
    """
    Reads a markdown file and returns its content as a string.

    :param file_path: Path to the markdown file.
    :return: Content of the markdown file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_json_file(dir, save_path):
    # iterate over all files in the directory
    for filename in os.listdir(dir):
        if filename.endswith(".md"):
            file_path = os.path.join(dir, filename)
            md_content = read_markdown_file(file_path)
            # Assuming you want to process the markdown content here
            llm = get_chat_model({
            'model': 'qwen3','model_server': 'http://10.24.9.6:11434/v1',
            'generate_cfg': {

                },
                })

            messages = [{'role': 'system', 'content': '你是文档总结专家，突出是vacc场景，结合文件名中的框架等信息，仅输出如下格式输出\{"summary":summray\}'},
                        {'role': 'user', 'content': f"{md_content}\n\n,文件名：{filename}，请结合文件名及内容，写下总结，长度30个字左右"}]

            responses = llm.chat(messages=messages, stream=False)
            # string to json
            import json
            response_json = json.loads(responses[0]['content'].split('</think>\n\n')[-1])
            print(response_json)
            # Save the response to a JSON file with the same name but .json extension
            # For example, let's just save it as a JSON with the same name but .json extension
            json_data = {'title': response_json['title']}
            json_file_path = os.path.join(save_path,filename[:-3] + ".json")
            print(f"Saving JSON to {json_file_path}")
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# Example usage
if __name__ == "__main__":
    dir_path = sys.argv[1]
    save_path = sys.argv[2]
    save_json_file(dir_path, save_path)

