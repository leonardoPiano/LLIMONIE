import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
from paths import TRAIN_PATH,TEST_PATH,OUTPUT_MODEL_PATH
import json
import argparse
import torch
from utils import prepare_dataset
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = (major_version >= 8)

model_ids = {"llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
             "anita-8b": "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
             "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
             "phi3_small": "microsoft/Phi-3-small-8k-instruct",
             "phi3_medium": "microsoft/Phi-3-medium-128k-instruct",
             "italia-9b":"sapienzanlp/modello-italia-9b"}
BS=4
max_seq_length=2048

if __name__=="__main__":

    train_data=json.load(open(TRAIN_PATH,"r"))
    val_data=json.load(open(TEST_PATH,"r"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the name of the model you want to train")
    parser.add_argument("--epoch",help="number of epochs")
    parser.add_argument("--unsloth", help="train with unsloth package",action='store_true')

    llm=None
    args = parser.parse_args()
    print(int(args.epoch))
    if args.model not in model_ids.keys():
        print("Your model is not valid!")
        print("This are the valid keys:",list(model_ids.keys()))
        exit(0)

    model_id=model_ids[args.model]
    use_unsloth=bool(args.unsloth)
    print("USE UNSLOTH",use_unsloth)
    if use_unsloth==True:
        from LLM.Unsloth import UnslothLLM
        llm=UnslothLLM(model_id,inference=False)
    else:
        from LLM.HF import LLM
        llm=LLM(model_id)

    ###PREPARE INSTRUCT DATASET
    train_df=prepare_dataset(train_data)
    val_df=prepare_dataset(val_data)

    
    train_hf = Dataset.from_pandas(train_df)
    val_hf = Dataset.from_pandas(val_df.sample(100, random_state=42))
    train_hf = train_hf.map(llm.formatting_prompts_func, batched=True, )
    val_hf = val_hf.map(llm.formatting_prompts_func, batched=True, )
    
    ###LOAD PEFT MODEL
    llm.model=llm.prepare_peft_model()

    # INIT TRAINER

    train_args = TrainingArguments(
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=int(args.epoch),
        max_steps=-1,
        learning_rate=2e-4,
        logging_steps=250,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="epoch",
        fp16=not SUPPORTS_BFLOAT16,
        bf16=SUPPORTS_BFLOAT16,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant",
        seed=3407,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        output_dir=f"outputs/{args.model}",
    )



    trainer = SFTTrainer(
        args=train_args,
        model=llm.model,
        tokenizer=llm.tokenizer,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2)

    trainer.train()

    trainer.save_model(f"{OUTPUT_MODEL_PATH}/{args.model}")
    llm.tokenizer.save_pretrained(f"{OUTPUT_MODEL_PATH}/{args.model}")






