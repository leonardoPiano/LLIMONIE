import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
from PromptTemplates.instruct_prompt_templates import  *
from LLM.Unsloth import UnslothLLM
from LLM.HF import LLM
from paths import TEST_PATH,OUTPUT_MODEL_PATH
import time
import json
import argparse

from tqdm import tqdm


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the name of the model you want to use")
    parser.add_argument("--unsloth",action='store_true')
    args = parser.parse_args()
    model_name=args.model
    model_path=f"{OUTPUT_MODEL_PATH}/{model_name}"
    out_path=f"Predictions/{model_name}/tuning_output.json"
    
    
    output = []
    start_id=0
    if os.path.exists(out_path):
        output=json.load(open(out_path,"r"))
        start_id=len(output)
        
    print(start_id)
    unsloth=args.unsloth
    if unsloth:
      llm=UnslothLLM(model_path,inference=True)
    else:
      llm=LLM(model_id=model_path,is_lora=True)
    dataset = json.load(open(TEST_PATH,"r"))

    TASKS = {"NER": NER, "RE": RE, "JOINT": JOINT}
    
    for id_, item in tqdm(enumerate(dataset[start_id:])):
        text = item["text"]
        category = item["category"]
        entities = item["entities"]
        result = {"text_id": id_+start_id, "category": category}
        for task_key, task in TASKS.items():

            USER = text
            if task_key == "RE":
                USER = f"\nEntities:{entities}\n" + text
            messages = [{"role": "system", "content": task},
                        {"role": "user", "content": USER}]

            start = time.time()
            try:
              answer = llm.generate(messages, max_new_tokens=512)
            except:
              answer = ""
               

            if task_key == "NER":
                entities_list = [x.strip() for x in answer.split(";")]
                result["NER"] = {"Entities": entities_list, "time": time.time() - start}
            else:
                triplets = [x.strip() for x in answer.split("\n")]
                result[task_key] = {"Triplets": triplets, "time": time.time() - start}

        output.append(result)

        with open(f"Predictions/{model_name}/tuning_output.json", "w") as f:
            json.dump(output, f, indent=2)


