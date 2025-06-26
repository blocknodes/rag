import openai
import random
import string
import time
import cohere

def generate_random_string(length):
    # Define the alphabet (A-Z)
    alphabet = [string.ascii_uppercase]  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(alphabet) for _ in range(length))

def run_embeding(client, model_uid,args):
    # Generate a random string of length 10
    random_string = generate_random_string(args['len'])
    # warm up
    for i in range(10):
        client.embeddings.create(
            model=model_uid,
            input=["random_string"]
        )
    start = time.time()
    for i in range(args['num_reqs']):
        client.embeddings.create(
            model=model_uid,
            input=["random_string"]
        )
    end = time.time()
    latency = (end - start) / args['num_reqs']
    print(f"Time taken for embeding {args['num_reqs']} reqs length {args['len']} latency is {latency} seconds per request")

def run_rerank(client, model_uid,args):
    # Generate a random string of length 10
    #model = client.get_model(model_uid)
    query = generate_random_string(args['qlen'])
    corpus = [
        generate_random_string(args['klen'])
    ]
    # warm up
    for i in range(10):
        client.rerank(
            model=model_uid,
            query=query,
            documents=corpus)
    start = time.time()
    for i in range(args['num_reqs']):
        client.rerank(
            model=model_uid,
            query=query,
            documents=corpus)
    end = time.time()
    latency = (end - start) / args['num_reqs']
    print(f"Time taken for rerank {args['num_reqs']} is {end - start} seconds, qeury length is {args['qlen']}, corpus length is {args['klen']} latency is {latency} seconds per request")


def run_llm(client, model_uid,args):
    # warm up
    for i in range(1):
        client.chat.completions.create(
            model=model_uid,
            messages=[
                {
                    "content": generate_random_string(args['len']),
                    "role": "user",
                }
            ],
            max_tokens=args['max_tokens'],
        )
    start = time.time()
    total_len=0
    for i in range(args['num_reqs']):
        response=client.chat.completions.create(
            model=model_uid,
            messages=[
                {
                    "content": generate_random_string(args['len']),
                    "role": "user",
                }
            ],
            max_tokens=args['max_tokens'],
        )
        total_len = total_len + len(response.choices[0].message.content)
        print(len(response.choices[0].message.content))
    end = time.time()
    latency = (end - start) * 1024 / total_len
    print(f"Time taken for llm {args['num_reqs']} reqs length is {args['len']}  is {end - start} seconds, latency is {latency} seconds per request")

def run(client, model_uid,args,type="embedding"):
    if type == "embedding":
        run_embeding(client, model_uid,args)
    elif type == "reranker":
        run_rerank(client, model_uid,args)
    elif type == "llm":
        run_llm(client, model_uid,args)
    else:
        raise ValueError(f"Unknown type: {type}")




def get_client(ip,port):
    return openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{ip}:{port}/v1"
    )

if __name__ == "__main__":
    run(cohere.Client(base_url="http://192.168.30.127:11225", api_key="sk-fake-key"), "qwen3-reranker",{'qlen':64,'klen':512,'num_reqs':10},type="reranker")
    client = get_client("192.168.30.127",'11224')
    run(client, "qwen3-embedding",{'len':512,'num_reqs':100},type="embedding")
    client = get_client("192.168.30.127",'11223')
    run(client, "safe-guard",{'len':512,'num_reqs':10,'max_tokens':8},type="llm")
    client = get_client("192.168.30.127",'11222')
    run(client, "prompt_model",{'len':512,'num_reqs':10,'max_tokens':16384-1024},type="llm")

