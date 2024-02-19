
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse

import torch

from tokenizers import Tokenizer
from models.modeling_progen import ProGenForCausalLM
from models.conotoxin_analysis import process_toxin_file



########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float32', torch_dtype=torch.float32, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample


def sample(device, model, tokenizer, context, max_length, min_length, num_return_sequences, pad_token_id, repetition_penalty, temp, top_p):


            with torch.no_grad():
                input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)

                tokens_batch = model.generate(
                        input_ids,
                        do_sample=True,
                        temperature=temp,
                        max_length=max_length,
                        min_length=min_length,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=pad_token_id,
                        repetition_penalty=repetition_penalty,
                        bad_words_ids=[[27],[18],[6],[24],[29]],# Prevent the generation of amino acids X(27)O(18)B(6)U(24)Z(29).
                        top_p=top_p
                )

                as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
                return tokenizer.decode_batch(as_lists(tokens_batch))


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)



########################################################################
# main


def main():

# (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ConoGen_checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'True'))
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--repetition_penalty', default=1.5, type=float) 
    args = parser.parse_args()


# (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'{args.model}'  

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False


# (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')


# (4) sample

    with print_time('sampling'):

        for p_value in range(10,0,-1):
            p = p_value / 10.0  # Set the value of p from 0.0 to 1.0 with a step of 0.1

            for t_value in range(10,0,-1): 
                t = t_value / 10.0  # Set the value of t from 0.0 to 1.0 with a step of 0.1

                completions = sample(
                    device=device,
                    model=model,
                    tokenizer=tokenizer,
                    context=args.context,
                    pad_token_id=tokenizer.encode('<|pad|>').ids[0],
                    num_return_sequences=args.num_samples,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    repetition_penalty=args.repetition_penalty,
                    temp=t,
                    top_p=p,
                )

                truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]

                print(args.context)
                filename = f"p_{p}_t_{t}.txt"
                with open(filename, "a+") as file:
                    file.write(f"toxin_sequence,length,scaffold,scaffold_type,cysteine_count")
                    file.write("\n")
                    for (i, truncation) in enumerate(truncations):
                        toxin_sequence, length, scaffold, scaffold_type, cysteine_count = process_toxin_file(truncation)
                        file.write(f"{toxin_sequence},{length},{scaffold},{scaffold_type},{cysteine_count}\n")



if __name__ == '__main__':
    main()
    print('done.')
