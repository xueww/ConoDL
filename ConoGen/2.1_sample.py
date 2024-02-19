
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


def sample(device, model, tokenizer, context, max_length, min_length, decoding_strategy, top_k, num_return_sequences, top_p, temp, pad_token_id, repetition_penalty, num_beams):

    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)

# Choose decoding strategy.

        if decoding_strategy == "top-p":
            tokens_batch = model.generate(
                input_ids,
                do_sample=True,
                temperature=temp,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                bad_words_ids=[[27],[18],[6],[24],[29]], # Prevent the generation of amino acids X(27)O(18)B(6)U(24)Z(29).
                top_p=top_p  
            )
        elif decoding_strategy == "beam":
            tokens_batch = model.generate(
                input_ids,
                do_sample=False, 
                temperature=temp,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_return_sequences=num_beams,  # num_return_sequences <= num_beams!
                early_stopping=True,  
                pad_token_id=pad_token_id
            )
        elif decoding_strategy == "top-k":
            tokens_batch = model.generate(
                input_ids,
                do_sample=True,
                temperature=temp,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                bad_words_ids=[[27],[18],[6],[24],[29]],  
                top_k=top_k  
            )
        elif decoding_strategy == "greedy":
            tokens_batch = model.generate(
                input_ids,
                do_sample=False, 
                temperature=temp,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=1,  
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty
            )
        elif decoding_strategy == "top-p-k":
            tokens_batch = model.generate(
                input_ids,
                do_sample=True,
                temperature=temp,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                bad_words_ids=[[27],[18],[6],[24],[29]],    
                top_p=top_p , 
                top_k=top_k  
            )
        else:
            raise ValueError("Invalid decoding strategy")

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
    parser.add_argument('--output_file', type=str, default='artificial_conotoxin.txt')
    parser.add_argument('--decoding_strategy', type=str, default='top-p',help='greedy,beam,top-p,top-k')
    parser.add_argument('--save_strategy', type=str, default='file', help='print,file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'True'))
    parser.add_argument('--p', type=float, default=0.80)
    parser.add_argument('--t', type=float, default=0.70)
    parser.add_argument('--max-length', type=int, default=100)
    parser.add_argument('--min-length', type=int, default=10) 
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--repetition_penalty', default=1.5, type=float) 
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--num_beams', type=int, default=5)
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
        completions = sample(
            device=device,
            model=model,
            tokenizer=tokenizer,
            context=args.context,
            pad_token_id=tokenizer.encode('<|pad|>').ids[0],
            num_return_sequences=args.num_samples,
            max_length=args.max_length,
            min_length=args.min_length,
            decoding_strategy=args.decoding_strategy, 
            temp=args.t,
            top_p=args.p,
            top_k=args.k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams
        )

        truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]

        print(args.context)


# Select output method.
        
        if args.save_strategy == "print":
            for (i, truncation) in enumerate(truncations):
                toxin_sequence, length, scaffold, scaffold_type, cysteine_count = process_toxin_file(truncation)
                print(f"{toxin_sequence},{length},{scaffold},{scaffold_type},{cysteine_count}\n")
        elif args.save_strategy == "file":
            with open(args.output_file, "a+") as file:
                file.write(f"toxin_sequence,scaffold,scaffold_type,cysteine_count")
                file.write("\n")
                for (i, truncation) in enumerate(truncations):
                    toxin_sequence, length, scaffold, scaffold_type, cysteine_count = process_toxin_file(truncation)
                    file.write(f"{toxin_sequence},{length},{scaffold},{scaffold_type},{cysteine_count}\n")
        else:
            raise ValueError("Invalid save strategy")



if __name__ == '__main__':
    main()
    print('done.')
