from dotenv import load_dotenv
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class UnslothLLM():
    def __init__(self, model_id,max_seq_length=2048,inference=True):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="cuda:0"
        )

        
        if inference:
            FastLanguageModel.for_inference(model)
        
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model

    def prepare_peft_model(self):
        return FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    def generate(self,messages,max_new_tokens=512):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(input_ids=inputs,pad_token_id=self.tokenizer.pad_token_id,max_new_tokens=max_new_tokens, temperature=0.0,    do_sample=False,use_cache=True)
        response = outputs[0][inputs.shape[-1]:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)
        return answer

    def formatting_prompts_func(self,examples):
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }














