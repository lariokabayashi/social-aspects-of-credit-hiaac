import re
import warnings
from typing import List
 
import torch
import transformers
import langchain
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    BitsAndBytesConfig,
)

#langchain.debug = true

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "tiiuae/falcon-7b"
 
quantization_config = BitsAndBytesConfig(load_in_8bit = True, llm_int8_threshold=200.0)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, device_map="cuda:0", quantization_config = quantization_config, 
)
model = model.eval()
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

generation_config = model.generation_config
generation_config.temperature = 0.5
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 500
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

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

generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria = stopping_criteria,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Chatbot tem a capacidade de lembrar o contexto da nossa conversa anterior enquanto aborda a questão atual
memory = ConversationBufferWindowMemory(
    memory_key="history", k=6, return_only_outputs=True
)

# Para garantir resultados limpos no chatbot
class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        return re.sub(user_pattern, "", text)
 
    @property
    def _type(self) -> str:
        return "output_parser"
    
while True: 
    template = """"
    Current History:{history}
    Question: {input}
    """.strip()

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        output_parser=CleanupOutputParser(),
        verbose=True,
    )

    text = input("Question: ")
    res = chain(text)
    print(res["response"])