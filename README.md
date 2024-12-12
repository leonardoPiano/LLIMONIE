# LLIMONIIE: Large Language Instructed Model for Open Named Information Extraction
This repository hosts the code, models, and dataset of the paper: " LLIMONIIE: Large Language Instructed Model
for Open Named Italian Information Extraction"
## Method Overview
<p align="center">
  <img src="pipeline.png"/>
</p>
 The input consists of a document or single sentence, along with an instruction. The output is a new sequence that includes
the named entities or triples related to the text, according to the chosen instruction.

## 🤖 Models
- [Lllama3-8b](https://huggingface.co/leopiano98/LLIMONIIE_anita-8b)
- [Anita-8b](https://huggingface.co/leopiano98/LLIMONIIE_llama3-8b)
- [phi3-mini](https://huggingface.co/leopiano98/LLIMONIIE_phi3-mini)
## 💻 Quick Start
### Setup conda environment
Install the unsloth package following the repo [guide](https://github.com/unslothai/unsloth?tab=readme-ov-file#conda-installation)
### Clone the repository
```bash
git clone https://github.com/leonardoPiano/LLIMONIE.git
```
### Run the generation
```python
from PromptTemplates.instruct_prompt_templates import  NER,RE,JOINT
from LLM.Unsloth import UnslothLLM
model_path="leopiano98/LLIMONIIE_anita-8b"

llimonie=UnslothLLM(model_path,inference=True)
task=NER
text="Alessandro Manzoni è considerato uno dei maggiori romanzieri italiani di tutti i tempi per il suo celebre romanzo I promessi sposi"
messages = [{"role": "system", "content": task},
                        {"role": "user", "content": text}]
output= llimonie.generate(messages, max_new_tokens=512)
#output: Alessandro Manzoni[Writer|Person]; I promessi sposi[Novel|Book]; italiani[Nationality|Ethnicity] 
```
## 📝 Dataset 
The LLIMONIIE dataset is  stored under the folder: "CROSS-DATASETS/cleaned/". The dataset contains a total of 10'000 documents, distributed among 5 categories (Music&Films, Politics, Science, Technology, and Literature). Each document is annotated with a set of named entities and open triplets, where each mention contains two types and has the form "mention[type|hypernym].  
