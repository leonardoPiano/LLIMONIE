import torch
from transformers import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_kbit_training
from peft import AutoPeftModelForCausalLM

from dotenv import load_dotenv
load_dotenv()

class LLM():
    def __init__(self,model_id,is_lora=False):
        self.model_id=model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,trust_remote_code=True,device_map="cuda:0")
        if not self.tokenizer.pad_token: 
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.padding_side = 'right'
            
        if is_lora:
            self.model=AutoPeftModelForCausalLM.from_pretrained(
                       model_id, 
                       low_cpu_mem_usage=True,
                       torch_dtype=torch.float16,
                       load_in_4bit=False).to("cuda:0")
        else:
            self.model = AutoModelForCausalLM.\
                from_pretrained(self.model_id,quantization_config=nf4_config,trust_remote_code=True,device_map="cuda:0")
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_peft_model(self):
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        base_model = prepare_model_for_kbit_training(self.model)

        peft_model = get_peft_model(base_model,
                                    lora_config)

        peft_model.config.use_cache = False
        return peft_model

    def formatting_prompts_func(self, examples):
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                 convos]
        return {"text": texts, }

    def generate(self,messages,max_new_tokens=512,temperature=0.0):

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)
        if "Llama-3"  in self.model_id:
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=False,
                temperature=temperature,
                top_p=1,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
        else:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                top_p=1,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True)
        response = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response,skip_special_tokens=True)
        return answer