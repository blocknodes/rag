import os
import pandas as pd
from qwen_agent.llm import get_chat_model
import sys

# read csv file
df = pd.read_csv('test_with_context_same.csv')
# print first 5 rows
print(df.head())
#loop over each row
for index, row in df.iterrows():
    print(f"Processing row {index}")
    # extract user_input and response from the row
    if str(row['context']) == 'nan':
        continue
    print('###########')
    answer = row['answer']
    question = row['question']
    context = row['context']
    llm = get_chat_model({
        'model': sys.argv[1],
        'model_server': 'http://10.24.9.6:11434/v1'
    })
    messages = [
        {"role": "system", "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."},
        {"role": "user", "content": f'Question: {question}\nContext: {context}'}
    ]
    responses = llm.chat(messages,stream=False,)
    llm_cot=responses[0]['content'].split('</think>\n\n')[0]
    llm_anwser=responses[0]['content'].split('</think>\n\n')[-1]
    #row['llm_anwser']=llm_anwser
    df.at[index, 'llm_cot'] = llm_cot
    df.at[index, 'llm_anwser'] = llm_anwser

    llm = get_chat_model({
        'model': 'qwen3',
        'model_server': 'http://10.24.9.6:11434/v1'
    })

    messages = [
        {"role": "system", "content": "你是评分专家，给你两段文字，请根据一致程度打分，打分范围0到100，仅输出分数"},
        {"role": "user", "content": f'第一段文字: {answer}\'第二段文字:: {llm_anwser}'}
    ]
    responses = llm.chat(messages,stream=False,)
    print(f"Generated Response: {responses[0]}")
    try:
        llm_score=int(responses[0]['content'].split('</think>\n\n')[-1])
    except:
        print(f'!!!!no socre')
        continue
    #row['llm_score']=llm_score
    df.at[index, 'llm_score'] = llm_score

      # Print the generated response
df.to_csv(f'test_with_score_{sys.argv[1]}.csv', index=False)