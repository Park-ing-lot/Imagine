import argparse
import json
import logging
import os
import re

import numpy as np
import torch
import tqdm
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel_Imagine
from transformers import AutoTokenizer

import adapters
from PIL import Image

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_PATH = '/home/user16/HT/visually-augmented/tasks'


class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass


class PiqaInstanceReader(InstanceReader):
    """
    Reads the PIQA dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ""
        question = fields["goal"]
        label = fields.get('label', None)
        choices = [fields["sol1"], fields["sol2"]]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices


class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, and choices.
    """

    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
            "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
            "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
            "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
            "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
            "What will (.*) do next?": r"[SUBJ] then",
            "How would (.*) feel after?": r"[SUBJ] then",
            "How would you describe (.*)?": r"[SUBJ] is seen as",
            "What kind of person is (.*)?": r"[SUBJ] is seen as",
            "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
            "Why did (.*) do that?": r"Before, [SUBJ] wanted",
            "Why did (.*) do this?": r"Before, [SUBJ] wanted",
            "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
            "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
            "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
            "What will happen to (.*)?": r"[SUBJ] then",
            "What will happen to (.*) next?": r"[SUBJ] then"
        }

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        label = ord(label) - 65
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                subj = m.group(1)
                if subj.endswith('?'):
                    subj = subj[:-1]
                answer_prefix = ans_prefix.replace("[SUBJ]", subj)
                break

        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")

        choices = [
            " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
                "?", "").replace("wanted to wanted to", "wanted to").replace(
                "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

        context_with_choices = [f"{context} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices


class ATOMICInstanceReader(InstanceReader):
    """
    Reads the ATOMIC dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        question = fields['context']
        label = fields['correct']
        choices = [fields['candidates'][0], fields['candidates'][1], fields['candidates'][2]]
        return '', question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices


class CWWVInstanceReader(InstanceReader):
    """
    Reads the CWWV dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        question = fields['question']['stem']
        if question.endswith('.'):
            question = question[:-1]
        if not question.endswith('[MASK]'):
            print('should not happen')
            exit(0)
        question = question[:-7]
        label = ['A', 'B', 'C'].index(fields['answerKey'])
        choices = [fields['question']['choices'][0]['text'] + '.', fields['question']['choices'][1]['text'] + '.',
                   fields['question']['choices'][2]['text'] + '.']
        return '', question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return context, question, label, choices, context_with_choices


class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."

        label = fields['answer']
        choices = [fields['option1'], fields['option2']]
        label = int(label) - 1
        question = ''
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [context.replace("_", choice) for choice in choices]
        return context, question, label, choices, context_with_choices


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ''

        question = 'Q: ' + fields['question']['stem']
        label = ['A', 'B', 'C', 'D', 'E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = ['A: ' + c['text'] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice[0].lower() + choice[1:]}" for choice in choices]
        return context, question, label, choices, context_with_choices

class ANLIInstanceReader(InstanceReader):
    """
    Reads the aNLI dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        question = fields['context']
        label = ['A', 'B'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['statement'] for c in fields['statements']]
        return label, choices, question

    @overrides
    def fields_to_instance(self, fields):
        label, choices, question = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return '', question, label, choices, choices, context_with_choices
    
class QASCInstanceReader(InstanceReader):
    """
    Reads the QASC dataset into a unified format with context, question, label, and choices.
    """

    @overrides
    def to_uniform_fields(self, fields):
        context = ''
        question = 'Q: ' + fields['question']['stem']
        label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = ['A: ' + c['text'][0].lower() + c['text'][1:] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return '', question, label, choices, context_with_choices
    
class ARCInstanceReader(InstanceReader):

    @overrides
    def to_uniform_fields(self, fields):
        answer_key = {'A': 0, 'B': 1,  'C': 2,  'D': 3, 'E': 4,
                      '1': 0, '2': 1,  '3': 2,  '4': 3, '5': 4}
        context = ''
        question = 'Q: ' + fields['question']['stem']
        label = answer_key[fields['answerKey']]
        choices = ['A: ' + c['text'][0].lower() + c['text'][1:] for c in fields['question']['choices']]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return '', question, label, choices, context_with_choices
    
class SCIQInstanceReader(InstanceReader):

    @overrides
    def to_uniform_fields(self, fields):
        choices_tmp = [fields['distractor1'], fields['distractor2'], fields['distractor3'], fields['correct_answer']]
        context = ''
        question = 'Q: ' + fields['question']
        label = 3
        choices = ['A: ' + c[0].lower() + c[1:] for c in choices_tmp]
        return context, question, label, choices

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        context_with_choices = [f"{question} {choice}" for choice in choices]
        return '', question, label, choices, context_with_choices


INSTANCE_READERS = {"socialiqa": SocialIQAInstanceReader,
                    "winogrande": WinograndeInstanceReader,
                    "piqa": PiqaInstanceReader,
                    "commonsenseqa": CommonsenseqaInstanceReader,
                    "anli": ANLIInstanceReader,
                    'atomic': ATOMICInstanceReader,
                    'cwwv': CWWVInstanceReader,
                    'arc-easy': ARCInstanceReader,
                    'arc-challenge': ARCInstanceReader,
                    'qasc': QASCInstanceReader,
                    'sciq': SCIQInstanceReader}

@torch.no_grad
def load_img_feature(img_fn, processor, model):
    image = Image.open(img_fn).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    return outputs.last_hidden_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="gpt2-large", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--cache_dir", default=None, type=str, required=False, help="where the model is cached")
    parser.add_argument("--reader", default=None, type=str, required=True, help="which reader to use")
    args = parser.parse_args()
    logger.info(args)

    task = args.reader
    if args.lm != 'gpt2-large':
        model_path = ['gpt2'] + args.lm.split('/')[-1:] + [task]
        model_path = '_'.join([m for m in model_path if m != ''])
        out_dir = os.path.join(args.out_dir, model_path)
    else:
        out_dir = os.path.join(args.out_dir, 'gpt2_' + task)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file_image = os.path.join(out_dir, '{}_predictions_image.jsonl'.format(args.reader))
    out_file_text = os.path.join(out_dir, '{}_predictions_text.jsonl'.format(args.reader))
    out_file_all = os.path.join(out_dir, '{}_predictions_all.jsonl'.format(args.reader))
    log_file_image = os.path.join(out_dir, '{}_results_image.txt'.format(args.reader))
    log_file_text = os.path.join(out_dir, '{}_results_text.txt'.format(args.reader))
    log_file_all = os.path.join(out_dir, '{}_results_all.txt'.format(args.reader))

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device, args.cache_dir)

    # Load the dataset
    instance_reader = INSTANCE_READERS[args.reader]()

    # Load the vision-langugae model
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    image_model.eval()
    image_model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Load the dataset
    instance_reader = INSTANCE_READERS[args.reader]()

    gold = []
    predictions_image = []
    loss_image = []
    results_image = []
    predictions_text = []
    results_text = []
    loss_text = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    sample_id = 0
    with open(out_file_image, "w") as f_out_image:
        with open(out_file_text, "w") as f_out_text:
            with open(args.dataset_file) as f_in:
                for line in tqdm.tqdm(f_in):
                    fields_image = json.loads(line.strip())
                    fields_text = json.loads(line.strip())
                    context, question, label, choices, context_with_choices = instance_reader.fields_to_instance(fields_image)
                    if sample_id == 0:
                        results_text.append(json.dumps(context_with_choices))
                    gold.append(label)
                    # Tokenize and pad
                    tokenized = [tokenizer.encode(text) for text in context_with_choices]
                    max_length = max([len(text) for text in tokenized])
                    att_mask = torch.zeros((len(tokenized), max_length)).to(device)
                    for i in range(len(tokenized)):
                        att_mask[i][:len(tokenized[i])] = 1
                    tokenized = [text + [pad_token_id] * (max_length - len(text)) for text in tokenized]
                    tokenized = torch.tensor(tokenized).long().to(device)

                    ###
                    image_hidden_states = load_img_feature(os.path.join(IMG_PATH, args.reader, f'{sample_id}.png'), processor, image_model)
                    loss, itm, all_weights = get_lm_score(model, 
                                                          torch.stack([tokenized, tokenized]).reshape(-1, tokenized.shape[-1]), 
                                                          pad_token_id, 
                                                          torch.stack([att_mask, att_mask]).reshape(-1, att_mask.shape[-1]),
                                                          torch.stack([image_hidden_states.squeeze() for _ in range(len(context_with_choices))]))

                    prediction_t = int(np.argmin(loss))
                    prediction_i = int(np.argmin(itm))
                    
                    fields_image["prediction"] = prediction_i
                    fields_image['scores'] = itm.tolist()
                    predictions_image.append(prediction_i)
                    loss_image.append(itm.tolist())
                    f_out_image.write(json.dumps(fields_image) + "\n")

                    fields_text["prediction"] = prediction_t
                    fields_text['scores'] = loss.tolist()
                    predictions_text.append(prediction_t)
                    loss_text.append(loss.tolist())
                    f_out_text.write(json.dumps(fields_text) + "\n")

                    sample_id += 1

    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = (np.array(gold) == np.array(predictions_image)).mean()
        print(f"Accuracy: {accuracy:.3f}")
        results_image.append(f"Accuracy: {accuracy:.3f}")
    with open(log_file_image, 'w') as fout:
        for line in results_image:
            fout.write(line + '\n')

    if None not in gold:
        accuracy = (np.array(gold) == np.array(predictions_text)).mean()
        print(f"Accuracy: {accuracy:.3f}")
        results_text.append(f"Accuracy: {accuracy:.3f}")
    with open(log_file_text, 'w') as fout:
        for line in results_text:
            fout.write(line + '\n')

    score_image = [(torch.tensor(x) * -1) for x in loss_image]
    score_text = [(torch.tensor(x) * -1) for x in loss_text]

    # ensemble
    results_all = []
    best_acc = 0
    best_weight = 0
    predictions_all = []
    scores_all = None
    for i in range(100):
        weight = 0.01 * (i+1)
        tmp = []
        predictions = []
        for i, x in enumerate(score_image):
            ttmp = x.softmax(-1) * weight + score_text[i].softmax(-1) * (1-weight)
            tmp.append(ttmp.tolist())
            predictions.append(torch.argmax(ttmp, -1).item())

        accuracy = (np.array(gold) == np.array(predictions)).mean()

        if accuracy > best_acc:
            best_weight = weight
            best_acc = accuracy
            predictions_all = predictions
            scores_all = tmp
    
    print(f"Accuracy: {best_acc:.3f} | Weight: {best_weight} (Softmax)")
    results_all.append(f"Accuracy: {best_acc:.3f} | Weight: {best_weight} (Softmax)")

    tmp = []
    predictions = []
    for i, x in enumerate(score_image):
        ttmp = x + score_text[i]
        tmp.append(ttmp.tolist())
        predictions.append(torch.argmax(ttmp, -1).item())
    accuracy = (np.array(gold) == np.array(predictions)).mean()
    print(f"Accuracy: {accuracy:.3f} (Sum)")
    results_all.append(f"Accuracy: {accuracy:.3f} (Sum)")

    tmp = []
    predictions = []
    for i, x in enumerate(score_image):
        ttmp = x.softmax(-1) + score_text[i].softmax(-1)
        tmp.append(ttmp.tolist())
        predictions.append(torch.argmax(ttmp, -1).item())
    accuracy = (np.array(gold) == np.array(predictions)).mean()
    print(f"Accuracy: {accuracy:.3f} (Sum_softmax)")
    results_all.append(f"Accuracy: {accuracy:.3f} (Sum_softmax)")

    
    sample_id = 0
    with open(out_file_all, "w") as f_out:
        with open(args.dataset_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                context, question, label, choices, context_with_choices = \
                        instance_reader.fields_to_instance(fields_image)
                # gold.append(label)
                if sample_id == 0:
                    results_all.append(json.dumps(context))
                    results_all.append(json.dumps(question))
                    results_all.append(json.dumps(choices))
                fields["prediction"] = predictions_all[sample_id]
                fields['scores'] = scores_all[sample_id]
                f_out.write(json.dumps(fields) + "\n")
                sample_id += 1
            
    with open(log_file_all, 'w') as fout:
        for line in results_all:
            fout.write(line + '\n')


def get_lm_score(model, batch, pad_token_id, att_mask, image_hidden_states):
    """
    Get the cross entropy loss of the texts in batch using the langage model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        model.active_adapters = adapters.composition.BatchSplit("mlm_expert", "itm_expert",
                                                                batch_sizes=[num_choices-len(image_hidden_states), len(image_hidden_states)])
        # print(image_hidden_states.shape, batch.shape, att_mask.shape)
        # torch.Size([3, 257, 1024]) torch.Size([6, 25]) torch.Size([3, 25])
        shift_labels = batch[:-len(image_hidden_states), 1:].contiguous().view(-1)
        outputs = model(batch, attention_mask=att_mask, image_hidden_states=image_hidden_states)
        lm_logits = outputs[0]
        attention_weights = outputs[-2]
        itm = outputs[-1]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_choices-len(image_hidden_states), -1).sum(1).cpu().numpy()
        valid_tokens = (batch != pad_token_id).long().sum(1).cpu().numpy()
        loss /= valid_tokens[:-len(image_hidden_states)]
    return loss, -itm.view(-1).cpu().numpy(), attention_weights


def init_model(model_name: str,
               device: torch.device, cache_dir):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = GPT2LMHeadModel_ROMI.from_pretrained(model_name, cache_dir=cache_dir)
    ###
    adapters.init(model) 
    adapter_name1 = model.load_adapter(f'{model_name}/adapter_mlm')
    adapter_name2 = model.load_adapter(f'{model_name}/adapter_itm')
    model.set_active_adapters([adapter_name1, adapter_name2])

    print(f'Initialized with {adapter_name1} and {adapter_name2} !!!')

    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    main()
