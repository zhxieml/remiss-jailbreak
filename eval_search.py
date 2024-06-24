from collections import defaultdict
from typing import List

import hydra
import numpy as np
import pandas as pd
import setproctitle
import torch
from omegaconf import DictConfig
from tqdm import trange

from src.advprompteropt import advPrompterOpt
from src.llm import LLM
from src.remissopt import reMissOpt
from src.sequence import Seq
from src.utils import check_jailbroken, get_test_prefixes

setproctitle.setproctitle("main")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_data(data_path):
    df = pd.read_csv(data_path)
    user_prompt = df.instruct.to_list()
    target = df.target.to_list()

    return user_prompt, target

class SuffixOpt:
    def __init__(
        self,
        cfg,
        target_llm,
        attack_llm,
    ):
        self.cfg = cfg
        self.target_llm = target_llm
        self.attack_llm = attack_llm

    def optimize(self, user_prompt: Seq, target: Seq) -> List[str]:
        if self.cfg.train.opt_type == "remiss":
            suffix, stats = reMissOpt(
                cfg=self.cfg,
                instruct=user_prompt,
                target=target,
                prompter=self.attack_llm,
                target_llm=self.target_llm,
                record_stats=True,
            )
        elif self.cfg.train.opt_type == "advprompter":
            suffix, stats = advPrompterOpt(
                cfg=self.cfg,
                instruct=user_prompt,
                target=target,
                prompter=self.attack_llm,
                target_llm=self.target_llm,
                record_stats=True,
            )
        else:
            raise ValueError(f"Invalid opt_type: {self.cfg.train.opt_type}")

        return (
            suffix,
            self.target_llm.tokenizer.batch_decode(
                torch.concat([user_prompt.ids, suffix.ids], dim=1),
                skip_special_tokens=True,
            ),
            stats,
        )

    def evaluate_perplexity(self, user_prompt: Seq, suffix: Seq):
        basemodel_tf = self.attack_llm.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=user_prompt,
            suffix=suffix,
            use_basemodel=True,
            loss_params=dict(hard_labels=True),
        )
        return basemodel_tf.perplexity


@hydra.main(version_base=None, config_path="conf_test")
def main(cfg: DictConfig):
    # load data
    user_prompt, target = load_data(cfg.eval.data.dataset_pth_dct.test)

    # set seed
    set_seed(cfg.seed)

    # load test prefixes
    test_prefixes = get_test_prefixes()

    # load models
    target_llm = LLM(cfg.target_llm)
    attack_llm = LLM(cfg.prompter) if cfg.prompter is not None else None
    suffix_opt = SuffixOpt(cfg, target_llm, attack_llm)

    # optimize
    all_res = defaultdict(list)
    for i in trange(0, len(target), cfg.train.batch_size):
        batch_user_prompt = user_prompt[i : i + cfg.train.batch_size]
        batch_target = target[i : i + cfg.train.batch_size]
        batch_user_prompt = Seq(
            text=batch_user_prompt,
            tokenizer=attack_llm.tokenizer,
            device=attack_llm.device,
        )
        batch_target = Seq(
            text=batch_target,
            tokenizer=target_llm.tokenizer,
            device=target_llm.device,
        )

        if cfg.train.add_target_whitespace:
            batch_target_optimize = Seq(
                text=[" " + t for t in batch_target.text],
                tokenizer=target_llm.tokenizer,
                device=target_llm.device,
            )
            suffices, full_instruct_text, batch_stats = suffix_opt.optimize(
                batch_user_prompt, batch_target_optimize
            )
        else:
            suffices, full_instruct_text, batch_stats = suffix_opt.optimize(
                batch_user_prompt, batch_target
            )
        full_instruct = Seq(
            text=full_instruct_text,
            tokenizer=target_llm.tokenizer,
            device=target_llm.device,
        )

        # evaluate 1: loss
        target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=batch_target,
            loss_params=dict(
                hard_labels=True,
                reweight_loss=cfg.reweight_loss,
            ),
        )
        batch_losses = target_llm_tf.loss_batch
        batch_ppls = suffix_opt.evaluate_perplexity(batch_user_prompt, suffices)

        # evaluate 2: asr
        target_llm_ar = target_llm.generate_autoregressive(
            key="target",
            full_instruct=full_instruct,
        )
        _, batch_jailbrokens = check_jailbroken(
            seq=target_llm_ar.response_sample, test_prefixes=test_prefixes
        )

        all_res["forbidden_prompt"].extend(batch_user_prompt.text)
        all_res["response"].extend(target_llm_ar.response_sample.text)
        all_res["suffix"].extend(suffices.text)
        all_res["loss"].append(batch_losses)
        all_res["perplexity"].append(batch_ppls)
        all_res["jailbroken"].extend(batch_jailbrokens)
        print(f"perplexity: {batch_ppls}")
        print(f"jailbroken: {batch_jailbrokens}")
        for k, v in batch_stats.items():
            all_res[k].append(v)
            print(f"{k}: {v}")

    for k, v in all_res.items():
        try:
            all_res[k] = torch.cat(v).cpu().numpy().tolist()
            print(f"{k}: {np.mean(all_res[k])}")
        except:
            pass

    df = pd.DataFrame(dict(all_res))
    output_path = cfg.output_dir + "/batch_results.jsonl"
    df.to_json(output_path, orient="records", lines=True)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
