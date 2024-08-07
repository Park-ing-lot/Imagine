import json
import pandas as pd  
from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm  
import shutil
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True)

args = parser.parse_args()

if args.task_name not in ['piqa', 'winogrande', 'anli', 'socialiqa', 'commonsenseqa', 'arc-easy', 'arc-challenge', 'qasc', 'sciq']:
    print(f'{args.task_name} not supported.')
    raise


'''
{"id": "58090d3f-8a91-4c89-83ef-2b4994de9d241", "context": "Ron started his new job as a landscaper today.", "question": {"stem": "Ron is immediately fired for insubordination.", "choices": [{"label": "A", "text": "Ron ignores his bosses's orders and called him an idiot."}, {"label": "B", "text": "Ron's boss called him an idiot."}]}, "statements": [{"label": true, "statement": "Ron started his new job as a landscaper today. Ron ignores his bosses's orders and called him an idiot. Ron is immediately fired for insubordination."}, {"label": false, "statement": "Ron started his new job as a landscaper today. Ron's boss called him an idiot. Ron is immediately fired for insubordination."}], "answerKey": "A"}
{"answerKey": "A", "id": "e68fb2448fd74e402aae9982aa76e527", "question": {"question_concept": "hamburger", "choices": [{"label": "A", "text": "fast food restaurant"}, {"label": "B", "text": "pizza"}, {"label": "C", "text": "ground up dead cows"}, {"label": "D", "text": "mouth"}, {"label": "E", "text": "cow carcus"}], "stem": "Where are  you likely to find a hamburger?"}}
{"context": "Carson was excited to wake up to attend school.", "question": "Why did Carson do this?", "answerA": "Take the big test", "answerB": "Just say hello to friends", "answerC": "go to bed early", "correct": "B"}
{"goal": "dresser", "sol1": "replace drawer with bobby pin ", "sol2": "finish, woodgrain with  bobby pin ", "label": 1}
{"qID": "3FCO4VKOZ4BJQ6IFC0VAIBK4KTWE7U-2", "sentence": "Sarah was a much better surgeon than Maria so _ always got the easier cases.", "option1": "Sarah", "option2": "Maria", "answer": "2", "ed": "1", "wid": "A1EPE2IRWQ9XO2", "ctx": "surgeon", "url": "https://www.wikihow.com/Become-an-Orthopedic-Surgeon", "domain": "social", "gender": "f", "meta_gender": "3"}
'''


if args.task_name == 'anli':
    file_path = 'anli_dev.jsonl'
    image_path = 'anli'
    task_name ='anli'
elif args.task_name == 'piqa':
    file_path = 'piqa_dev.jsonl'
    image_path = 'piqa'
    task_name ='piqa'
elif args.task_name == 'socialiqa':
    file_path = 'socialiqa_dev.jsonl'
    image_path = 'socialiqa'
    task_name ='socialiqa'
elif args.task_name == 'winogrande':
    file_path = 'winogrande_dev.jsonl'
    image_path = 'winogrande'
    task_name ='winogrande'
elif args.task_name == 'arc-easy':
    file_path = 'arc-easy_dev.jsonl'
    image_path = 'arc-easy'
    task_name ='arc-easy'
elif args.task_name == 'arc-challenge':
    file_path = 'arc-challenge_dev.jsonl'
    image_path = 'arc-challenge'
    task_name ='arc-challenge'
elif args.task_name == 'qasc':
    file_path = 'qasc_dev.jsonl'
    image_path = 'qasc'
    task_name ='qasc'
elif args.task_name == 'sciq':
    file_path = 'sciq_dev.jsonl'
    image_path = 'sciq'
    task_name ='sciq'
else:
    file_path = 'commonsenseqa_dev.jsonl'
    image_path = 'commonsenseqa'
    task_name ='commonsenseqa'

data = []
with open(file_path) as f:
    for line in f:
        data.append(json.loads(line))
        

def get_context(data, task_name):
    contexts = []
    for d in data:
        if task_name == 'anli':
            context = ' '.join([d['context'], d['question']['stem']])
            contexts.append(context)
        elif task_name == 'piqa':
            contexts.append(d['goal'])
        elif task_name == 'socialiqa':
            context = ' '.join([d['context'], d['question']])
            contexts.append(context)
        elif ('arc' in task_name) or ('qasc' in task_name):
            context = d['question']['stem']
            contexts.append(context)
        elif task_name == 'sciq':
            context = d['question']
            contexts.append(context)
        elif task_name == 'winogrande':
            tmp = []
            for w in d['sentence'].split():
                if w == '_': break
                tmp.append(w)
            context = ' '.join(tmp)
            contexts.append(context)
        else:
            contexts.append(d['question']['stem'])
    
    return contexts


contexts = get_context(data, task_name)


pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/juggernaut-xl-v5",
    torch_dtype=torch.float16,
    allow_pickle=False
    )

pipeline.load_lora_weights("openskyml/dalle-3-xl", use_safetensors=True)
pipeline.set_progress_bar_config(disable=True)
pipeline.to('cuda')


for i, prompt in enumerate(tqdm(contexts)):
    with torch.no_grad():
        output = pipeline(prompt, 
                        height=512,
                        width=512,
                        num_inference_steps=50,
                        num_images_per_prompt=1,
                        # generator=generator
                        )
    for image in output.images:
        image.save(f'{image_path}/{i}.png')
