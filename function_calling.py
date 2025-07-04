# Reference: https://platform.openai.com/docs/guides/function-calling
import json
import os
from ragflow_sdk import RAGFlow
from qwen_agent.llm import get_chat_model
import openai
import numpy as np
from tqdm import tqdm

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

# add args using argparse
import argparse
# add args for source_files and query
parser = argparse.ArgumentParser(description='Rag for Vacc Model Zoo')
parser.add_argument('--source_files', type=str, nargs='+',
                    help='a list of source files')
parser.add_argument('--query', type=str,
                    help='the query to retrieve information')

parser.add_argument('--online-index-dir', type=str,
                    help='the query to retrieve information')

parser.add_argument('--online-orig-dir', type=str,
                    help='the query to retrieve information')

parser.add_argument('--topk', type=int,
                    help='the query to retrieve information')

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def retrieve(query, sources):
    """Retrieve information based on a query and sources"""
    print(f"Retrieved information for {query} from {sources}")
    contents = []
    for file_path in sources:
        with open(file_path, 'r', encoding='utf-8') as file:
           contents.append(file.read())

    return contents

def retrieve_from_ragflow(query,top_k=4):
    rag_object = RAGFlow(api_key="ragflow-A1M2ViY2I0MjNkNjExZjA5NzEzODJjYm", base_url="http://10.24.73.27:8080")
    dataset = rag_object.list_datasets(name="modelzoo_base")
    dataset = dataset[0]

    result=[]
    for c in rag_object.retrieve(question=query, dataset_ids=[dataset.id],rerank_id="bge-reranker-v2-m3@Xinference",vector_similarity_weight=0.9):
        result.append(c.content)
    return result[:top_k]

def embedding(text, model="bge"):
    """
    Generates an embedding for the given text using OpenAI's API.
    :param text: The input text to generate an embedding for.
    :return: Embedding vector as a list of floats.
    """
    client = openai.OpenAI(base_url='http://10.24.9.6:19875/v1',api_key='EMPTY')

    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.dict()['data'][0]['embedding']
    except OpenAIError as e:
        print(f"An error occurred: {e}")
        return None

def read_markdown_file(file_path):
    """
    Reads a markdown file and returns its content as a string.

    :param file_path: Path to the markdown file.
    :return: Content of the markdown file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def retrieve_online(query,index_dir,origin_dir, topk=2):
    query_embedding = embedding(query)
    lst1=[query_embedding]
    lst2 = []
    lst2_names=[]
    for file in tqdm(os.listdir(index_dir)):
        file_path = os.path.join(index_dir, file)
        title = json.loads(open(file_path).read())['title']
        title_embedding = embedding(title)
        lst2_names.append(file)
        lst2.append(np.array(title_embedding))
    lst1 = np.array(lst1)
    lst2 = np.array(lst2)
    result = lst1 @ lst2.T
    print(result)
    ## use top-k
    top_k_indices = np.argsort(result, axis=-1)[:, -topk:]
    print(top_k_indices)
    selected_sources = []
    selected_files = []
    for index in top_k_indices[0]:
        file_name = lst2_names[index]
        selected_files.append(file_name)
        file_path = os.path.join(origin_dir, file_name[:-4]+"md")

        md_content = read_markdown_file(file_path)
        md_content = f'文件名:{file_name[:-5]}\n文件内容:\n{md_content}'
        selected_sources.append(md_content)
    print(f'selected file is {selected_files}')


    return selected_sources

def format_response(sources):
    idx=1
    contents=[]
    for source in sources[::-1]:
        contents.append(f'source {idx}:\n{source}\n')
        idx = idx +1

    return "\n\n".join(contents)

def test(fncall_prompt_type: str = 'qwen'):
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        'model': 'qwen3',
        'model_server': 'http://10.24.9.6:11434/v1',
        'generate_cfg': {
            'fncall_prompt_type': fncall_prompt_type
        },
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'system', 'content': "你是瀚小博，由瀚博科技研发，你是一个model zoo知识库问答助手，你可以帮助用户查询model zoo中的相关信息，但请礼貌拒绝其他无关问题."},
                {'role': 'user', 'content': parser.parse_args().query}]
    functions = [{
        'name': 'retrieve',
        'description': '根据询问查询vacc的model zoo(vacc是一个机器学习平台框架)',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'query',
                },
            },
            'required': ['query'],
        },
    }]

    print('# Assistant Response 1:')
    responses = []
    for responses in llm.chat(
            messages=messages,
            functions=functions,
            stream=False,
            # Note: extra_generate_cfg is optional
            # extra_generate_cfg=dict(
            #     # Note: if function_choice='auto', let the model decide whether to call a function or not
            #     # function_choice='auto',  # 'auto' is the default if function_choice is not set
            #     # Note: set function_choice='get_current_weather' to force the model to call this function
            #     function_choice='get_current_weather',
            # ),
    ):
        print(responses)

    # If you do not need streaming output, you can either use the following trick:
    #   *_, responses = llm.chat(messages=messages, functions=functions, stream=True)
    # or use stream=False:
    #   responses = llm.chat(messages=messages, functions=functions, stream=False)

    messages.append(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    if last_response.get('function_call', None):

        # Step 3: call the function
        if parser.parse_args().source_files:
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                'retrieve': retrieve,
            }
        elif parser.parse_args().online_index_dir:
            available_functions = {
                'retrieve': retrieve_online,
            }
        else:
            available_functions = {
                'retrieve': retrieve_from_ragflow,
            }
        function_name = last_response['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response['function_call']['arguments'])
        if parser.parse_args().source_files:

            function_response = function_to_call(
                query=function_args.get('query'),
                sources=parser.parse_args().source_files,
            )
        elif parser.parse_args().online_index_dir:
            function_response = function_to_call(
                query=function_args.get('query'),
                index_dir=parser.parse_args().online_index_dir,
                origin_dir=parser.parse_args().online_orig_dir,
                topk=parser.parse_args().topk,
            )
        else:

            function_response = function_to_call(
                query=function_args.get('query'),
                top_k=parser.parse_args().topk,
            )

        # format
        function_response=format_response(function_response)
        print(f'{"#"*40}\n{function_response}')


        # Step 4: send the info for each function call and function response to the model
        hit_format=f'"ans":answer,"cita":number'
        miss_format = f'"ans":None,"cita":None'
        qeury_with_content=f'''{function_response}\n\n请依据以上内容回答，用户的问题，以json对象{hit_format}的形式输出,其中cita指的是引用的第几个资料,而不是具体资料中的第几个部分！,如果以上内容跟用户问题不相关，请输出{miss_format}\n,用户的问题是：{parser.parse_args().query}'''
        #print(qeury_with_content)
        messages = [{'role': 'system', 'content': "You are 瀚小博 for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."},
                   {'role': 'user', 'content': f'Context: {function_response}\n\nQuestion: {parser.parse_args().query}'}]

        print('# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                stream=False,
        ):  # get a new response from the model where it can see the function response
            print(responses['content'])


if __name__ == '__main__':
    # Run example of function calling with QwenFnCallPrompt
    # test(fncall_prompt_type='qwen')

    # Run example of function calling with NousFnCallPrompt
    test(fncall_prompt_type='nous')
