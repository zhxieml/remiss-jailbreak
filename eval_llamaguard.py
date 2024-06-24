import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaGuardEvaluator:
    def __init__(self, model_id="meta-llama/LlamaGuard-7b", device="cuda", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    def moderate(self, chat):
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = pd.read_csv(args.input).to_dict(orient="list")
    prompts, responses, grades = [], [], []
    evaluator = LlamaGuardEvaluator()

    for p, r in tqdm(zip(results["forbidden_prompt"], results["response"]), total=len(results)):
        res = evaluator.moderate([
            {"role": "user", "content": p},
            {"role": "assistant", "content": r.strip()},
        ])

        prompts.append(p)
        responses.append(r)
        grades.append(res)

    eval_results = pd.DataFrame(dict(
        results,
        grade=grades,
    ))
    eval_results.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="res/strongreject/tmp.csv")
    args = parser.parse_args()

    main(args)
