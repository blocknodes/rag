import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["OPENAI_BASE_URL"] = "http://10.24.9.6:11434/v1"

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="qwen3"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerAccuracy

sample = SingleTurnSample(
    user_input="When and where was Einstein born?",
    response="Albert Einstein was born in 1879, German.",
    reference="Albert Einstein was born in 1879."
)

scorer = AnswerAccuracy(llm=evaluator_llm) # evaluator_llm wrapped with ragas LLM Wrapper
score = scorer.single_turn_score(sample)
print(score)

#print(metric.single_turn_score(test_data))