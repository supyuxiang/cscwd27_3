from __future__ import annotations

import torch
from torch import nn

import json
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from generator.prompt import build_prompt


device = torch.device('cuda')




def get_pooling_fn(pooling_mode: str):

    assert pooling_mode in ['last', 'mean']
    
    def _last(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # hs: bsz, seq_len, d_model
        # attention_mask: bsz, seq_len
        last_indices = (attention_mask.sum(dim=-1, keepdim=False) -1).clamp(min=0) # bsz,
        return hidden_states[
            torch.arange(len(hidden_states)),
            last_indices,
        ]

    def _mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        hs = (hidden_states * attention_mask.unsqueeze(-1)).sum(dom=1) / attention_mask.sum(dim=-1)
        return hs

        
    return _last if pooling_mode == 'last' else _mean


def save_hs(hs, save_path):
    torch.save(hs, save_path)


def sampling(
    model: nn.Module,
    question_ls: list[str],
    res_ls: list[str],
    tokenizer,
    batch_size: int,
    save_dir: str,
    pooling_mode: str = 'last'
):
    prompt_ls = [
        (build_prompt(q) + res) for q, res in zip(question_ls, res_ls)
    ]

    hs_by_layer = {
        i:[] for i in range(1, model.config.num_hidden_layers + 1)
    }

    pooling_fn = get_pooling_fn(pooling_mode)

    @torch.inference_mode()
    def process_batch(batch):
        inputs = tokenizer(
            batch,
            padding=True,
            max_length=8192,
            tokenize=False,
            return_tensors='pt'
        )
        inputs = {k:v.to(device) for k,v in inputs.items()}
        attention_mask = inputs['attention_mask']

        hs = model(**inputs, output_hidden_states=True).hidden_states

        for layer in hs_by_layer:
            hs = hs[layer] # torch.Tensor, bsz, seq_len, d_model
            hs = pooling_fn(hidden_states=hs, attention_mask=attention_mask) # bsz, d_model
            hs_by_layer[layer].append(hs.copy().detach().cpu())
        del inputs, attention_mask, hs
        gc.collect()
        torch.cuda.empty_cache()

    for start in tqdm(range(0, len(prompt_ls), batch_size)):
        batch = prompt_ls[start: start+batch_size]
        process_batch(batch)


    # cat and save
    hs_by_layer = {
        layer: torch.cat(hs_by_layer[layer], dim=0)
        for layer in hs_by_layer
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for layer in hs_by_layer:
        hs = torch.cat(hs_by_layer[hs], dim=0)
        save_path = save_dir / f'layer{layer}.pt'
        save_hs(hs, save_path)
        del hs
        gc.collect()
        torch.cuda.empty_cache()
    






def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='')
    p.add_argument('--sampled_path', default='')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='')

    args = p.parse_args()

    with open(args.sampled_path,'r',encoding='utf-8') as f:
        sampled = json.load(f)
    question_ls, res_ls = [], []

    for item in sampled:
        question_ls.append(item['question'])
        res_ls.append(item['res'])

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_is is None:
        tokenizer.pad_token_is = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'right'
    

    sampling(
        model=model,
        question_ls=question_ls,
        res_ls=res_ls,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()