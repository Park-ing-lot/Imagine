import glob
import os

from tqdm import tqdm

eval_tasks = [
    ("socialiqa", "../../tasks/socialiqa_dev.jsonl"),
    ("winogrande", "../../tasks/winogrande_dev.jsonl"),
    ("piqa", "../../tasks/piqa_dev.jsonl"),
    ("commonsenseqa", "../../tasks/commonsenseqa_dev.jsonl"),
    ("anli", "../../tasks/anli_dev.jsonl")
]

total_models_to_eval = 0
for f in glob.glob('../Training/output'):
    for models in glob.glob('{}/gpt2*'.format(f)):
        if "gpt" not in models: continue
        total_models_to_eval += 1

progress_bar = tqdm(total=total_models_to_eval)

output_folders = glob.glob('../Training/output')
for f in output_folders:
    output_split = f.split('_')[-1]

    for models in glob.glob('{}/gpt2*'.format(f)):
        if "gpt" not in models: continue
        training_data = models.split('_')[-1]
        if not os.path.exists("./final_results/{}_{}gpt2-large".format(output_split, training_data)):
            for reader, dataset in eval_tasks:
                os.system(
                    """python evaluate_gpt_imagine.py --lm {} --dataset_file {} --out_dir {} --device 1 --reader {}""".format(
                        models, dataset, "./final_results/{}_{}_gpt2-large".format(output_split, training_data),
                        reader))
        progress_bar.update(1)