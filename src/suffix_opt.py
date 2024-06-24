from typing import List

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from advprompteropt import (
    advPrompterOpt,
    get_next_token_probabilities,
    select_and_evaluate_next_token_candidates,
    select_next_beams,
)
from sequence import MergedSeq, Seq, EmptySeq


class SuffixOpt:
    def __init__(
        self,
        target_llm,
        attack_llm,
    ):
        self.target_llm = target_llm
        self.attack_llm = attack_llm

    def optimize(self, user_prompt: List[str], target: List[str]) -> List[str]:
        raise NotImplementedError


class AdvPrompterOpt(SuffixOpt):
    @torch.no_grad()
    def draft(
        self,
        user_prompt: List[str],
        target: List[str],
        reweight_loss: bool,
        max_new_tokens: int,
        num_beams: int,
        verbose: bool,
    ) -> List[str]:
        if verbose:
            tqdm.write("\n Running AdvPrompterOpt: Generating optimized suffix...")

        # Compute the initial prediction losses without a suffix
        full_instruct = Seq(
            text=user_prompt,
            tokenizer=self.target_llm.tokenizer,
            device=self.target_llm.device,
        )
        target_llm_tf = self.target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=target,
            loss_params=dict(hard_labels=True, reweight_loss=reweight_loss),
        )
        losses = target_llm_tf.loss_batch.detach().to(self.attack_llm.device)  # (ebs, )

        # Initialize the beam scores
        beam_scores = torch.zeros_like(losses)  # (ebs, )
        suffix_beams = EmptySeq(
            tokenizer=self.attack_llm.tokenizer, device=self.attack_llm.device
        )

        for idx in range(max_new_tokens):
            if idx == 0:
                num_beams_in = 1
                num_beams_out = num_beams
            elif idx == max_new_tokens - 1:
                num_beams_in = num_beams
                num_beams_out = 1
            else:
                num_beams_in = num_beams
                num_beams_out = num_beams

            # expand the dimension of instruct and targets to match suffix beams
            instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
            target_rep = target.repeat_interleave(num_beams_in, dim=0)

            next_dist_seq_prompter, next_dist_seq_basemodel = (
                get_next_token_probabilities(
                    cfg=cfg,
                    instruct=instruct_rep,
                    suffix=suffix_beams,
                    prompter=self.attack_llm,
                )
            )

            next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
                select_and_evaluate_next_token_candidates(
                    cfg=cfg,
                    instruct=instruct_rep,
                    target=target_rep,
                    suffix=suffix_beams,
                    target_llm=self.target_llm,
                    next_dist_seq_prompter=next_dist_seq_prompter,
                    next_dist_seq_basemodel=next_dist_seq_basemodel,
                    beam_scores=beam_scores,
                    prev_losses=losses,
                    num_beams_in=num_beams_in,
                )
            )

            suffix_beams, losses, beam_scores = select_next_beams(
                cfg=cfg,
                suffix_beams=suffix_beams,
                next_token_candidate_ids=next_token_candidate_ids,
                candidate_beam_scores=candidate_beam_scores,
                candidate_losses=candidate_losses,
                num_beams_in=num_beams_in,
                num_beams_out=num_beams_out,
            )
            if verbose:
                tqdm.write(
                    f" Beams[0] (iter {idx}): {suffix_beams[:num_beams_out].text}"
                )

        if verbose:
            tqdm.write(
                f" AdvPrompterOpt completed. Generated suffix[0]: {suffix_beams[0].text}"
            )
        return suffix_beams

    def optimize(self, user_prompt: List[str], target: List[str]):
        cfg = DictConfig(
            dict(
                verbose=True,
                reweight_loss=True,
                train=dict(q_params=dict(max_new_tokens=20)),
            )
        )
        suffix = advPrompterOpt(
            cfg=cfg,
            instruct=user_prompt,
            target=target,
            prompter=self.attack_llm,
            target_llm=self.target_llm,
        )

        return suffix
