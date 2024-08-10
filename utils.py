import pandas as pd
import random
from PromptTemplates.instruct_prompt_templates import *


def normalize_triplet_entities(triplet):
    norm = []
    for item in triplet.split(";"):
        norm.append(item.strip())
    return ";".join(norm)


def prepare_dataset(json_dataset: list[dict]):
    dataset_df = pd.DataFrame(columns=["conversations"])

    for item in json_dataset:
        entities_list = item["entities"]
        text = item["text"]
        ent_response = "; ".join(entities_list)

        norm_triplets = [normalize_triplet_entities(x) for x in item["triplets"]]
        trip_response = "\n".join(norm_triplets)
        re_entities = []
        re_response = ""

        sample = random.randint(0, len(norm_triplets))
        trip_sample = random.sample(norm_triplets, sample)

        if len(trip_sample) > 0:
            re_response = "\n".join(trip_sample)
            for trip in trip_sample:
                s, p, o = trip.split(";")
                if s in entities_list and o in entities_list:
                    if s not in re_entities:
                        re_entities.append(s)
                    if o not in re_entities:
                        re_entities.append(o)
                if len(re_entities) == 0:
                    re_response = "[]"


        else:
            for ent in entities_list:
                if ent not in str(norm_triplets):
                    re_entities.append(ent)
            re_response = "[]"

            # re_response=trip_response

        ner_inst = NER
        re_inst = RE 
        joint_inst = JOINT
        instructions = [(ner_inst, ent_response), (re_inst, re_response), (joint_inst, trip_response)]

        for i, instruction in enumerate(instructions):
            USER = text
            if i == 1:
                USER = f"\nEntities:{re_entities}" + text
            dialog = [{"role": "system", "content": instruction[0]},
                      {"role": "user", "content": USER},
                      {"role": "assistant", "content": instruction[1]}]
    
            dataset_df.loc[len(dataset_df)] = {"conversations": dialog}


    return dataset_df

