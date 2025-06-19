#read csv file
import os
import pandas as pd
from ragflow_sdk import RAGFlow

def retrieve_from_ragflow(query,top_k=1):
    rag_object = RAGFlow(api_key="ragflow-A1M2ViY2I0MjNkNjExZjA5NzEzODJjYm", base_url="http://10.24.73.27:8080")
    dataset = rag_object.list_datasets(name="modelzoo_base")
    dataset = dataset[0]

    result=[]
    chunks = rag_object.retrieve(question=query,
        dataset_ids=[dataset.id],
        #rerank_id="bge-reranker-v2-m3@Xinference",
        vector_similarity_weight=0.7,
)
    for c in chunks[:top_k]:
        result.append(c.content)
    return result

# read csv file
df = pd.read_csv('test.csv')
# print first 5 rows
print(df.head())
#loop over each row
for index, row in df.iterrows():
    print(f"Processing row {index}")
    # extract user_input and response from the row
    answer = row['answer']
    question = row['question']
    reposonse = retrieve_from_ragflow(answer)
    reposonse_full = retrieve_from_ragflow(question+answer)
    if reposonse!=reposonse_full:
        print(f'answer:{answer}\n\n{reposonse}\n\n{reposonse_full}\n\n')
        continue
    df.at[index, 'context'] = reposonse
    # add to row

    print(f"Retrieved Answer: {reposonse}")  # Print the retrieved answer


df.to_csv('test_with_context_same.csv', index=False)