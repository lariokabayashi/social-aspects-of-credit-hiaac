from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, StoppingCriteriaList, StoppingCriteria, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import PromptTemplate,  LLMChain, ConversationChain
from typing import List
import torch
from langchain.llms import HuggingFacePipeline

model_name = "tiiuae/falcon-7b" 

tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(load_in_8bit = True, llm_int8_threshold=200.0)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", quantization_config = quantization_config).eval()
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, trust_remote_code=True, device_map="cuda:0", quantization_config = quantization_config, 
# )

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False
    
# stop_tokens = [["Human", ":"], ["AI", ":"], ["Question", ":"]]
stop_tokens = [["Answer", ":"],["Question", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # device_map="cuda:0",
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria = stopping_criteria
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

 

# template = """
# You are an intelligent chatbot. Help the following question with brilliant answers.
# Question: {question}
# Answer:"""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = input()

# print(llm_chain.run(question))
while True:
    template = """"
    Current History:{history}
    Question: {input}
    """.strip()

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    text = input("Question: ")
    res = chain(text)
    print(res["response"])