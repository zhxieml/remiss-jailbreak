import torch
from tqdm import tqdm

from src.advprompteropt import (
    get_next_token_probabilities,
    select_next_beams,
    select_next_token_candidates,
)
from src.sequence import EmptySeq, MergedSeq, Seq, stack_seqs


@torch.no_grad()
def reMissOpt(cfg, instruct, target, prompter, target_llm, record_stats=False):
    if cfg.verbose:
        tqdm.write("\n Running reMissOpt: Generating optimized suffix...")

    # Compute the initial losses without a suffix
    full_instruct = Seq(
        text=instruct.text, tokenizer=target_llm.tokenizer, device=target_llm.device
    )
    target_llm_ar = target_llm.generate_autoregressive(
        key="target",
        full_instruct=full_instruct,
        max_new_tokens=target.ids.shape[1],
    )
    baseline = target_llm_ar.response_sample

    if record_stats:  # for analysis
        target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )
        init_losses = losses.clone()

    # Truncate baselines to match the targets
    baseline_ids = baseline.ids
    baseline_ids[target.ids == target.tokenizer.pad_token_id] = (
        baseline.tokenizer.pad_token_id
    )
    baseline = Seq(
        ids=baseline_ids,
        tokenizer=baseline.tokenizer,
        device=baseline.device,
    )
    target = Seq(
        ids=target.ids,
        tokenizer=target.tokenizer,
        device=target.device,
    )

    # Calculate rewards
    all_rewards = calculate_reward(
        cfg,
        prompter=prompter,
        target_llm=target_llm,
        instruct=stack_seqs([instruct] * 2),
        suffix=EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device),
        target=stack_seqs([target, baseline]),
    )
    reward_gaps = all_rewards[: instruct.bs] - all_rewards[instruct.bs :]

    init_reward_gaps = reward_gaps.clone()

    # Initialize the beam scores
    beam_scores = torch.zeros_like(reward_gaps)  # (ebs, )
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    # Main loop: Generate the optimized suffix iteratively
    for idx in range(cfg.train.q_params.max_new_tokens):
        if idx == 0:
            num_beams_in = 1
            num_beams_out = cfg.train.q_params.num_beams
        elif idx == cfg.train.q_params.max_new_tokens - 1:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = 1
        else:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = cfg.train.q_params.num_beams

        # Expand the dimension of instruct and targets to match suffix beams
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)
        baseline_rep = baseline.repeat_interleave(num_beams_in, dim=0)

        # Get the next token probabilities as candidates
        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg, instruct=instruct_rep, suffix=suffix_beams, prompter=prompter
        )

        # Select and evaluate the next token candidates
        next_token_candidate_ids, candidate_beam_scores, candidate_reward_gaps = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                instruct=instruct_rep,
                target=target_rep,
                baseline=baseline_rep,
                suffix=suffix_beams,
                prompter=prompter,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=reward_gaps,
                num_beams_in=num_beams_in,
            )
        )

        # Select the next beams
        prev_reward_gaps = reward_gaps
        suffix_beams, reward_gaps, beam_scores = select_next_beams(
            cfg=cfg,
            suffix_beams=suffix_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_reward_gaps,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )

    if cfg.verbose:
        tqdm.write(f" reMissOpt completed. Generated suffix[0]: {suffix_beams[0].text}")

    if record_stats:
        target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=Seq(
                text=MergedSeq(seqs=[instruct, suffix_beams])
                .to_seq(merge_dtype="ids")
                .text,
                tokenizer=target_llm.tokenizer,
                device=target_llm.device,
            ),
            target=target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )

        return suffix_beams, dict(
            init_reward_gap=torch.tensor(init_reward_gaps),
            reward_gap=torch.tensor(reward_gaps),
        )
    return suffix_beams


def calculate_reward(cfg, prompter, target_llm, instruct, suffix, target):
    """Calculate log_target(y | x) - log_base(y | x) for the given x and y."""
    target_llm_instruct = Seq(
        text=MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target_llm_target = target

    prompter_instruct = instruct
    prompter_suffix = suffix
    prompter_target = Seq(
        text=target.text, tokenizer=prompter.tokenizer, device=prompter.device
    )

    if cfg.train.q_params.lambda_val:
        target_llm_losses = (
            target_llm.compute_pred_loss_teacher_forced(
                key="target",
                full_instruct=target_llm_instruct,
                target=target_llm_target,
                loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
            )
            .loss_batch.detach()
            .cpu()
        )
    else:
        target_llm_losses = torch.zeros((instruct.bs,))
    if cfg.train.q_params.use_prompter_reward:
        prompter_losses = (
            prompter.compute_pred_loss_teacher_forced(
                key="target",
                instruct=prompter_instruct,
                suffix=prompter_suffix,
                target=prompter_target,
                loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
                use_basemodel=True,  # NOTE: compare the results with and without this
            )
            .loss_batch.detach()
            .cpu()
        )
    else:
        prompter_losses = torch.zeros_like(target_llm_losses)

    return prompter_losses - cfg.train.q_params.lambda_val * target_llm_losses


def calculate_reward_v2(cfg, prompter, target_llm, instruct, suffix, target):
    """Calculate log_target(y | x) - log_base(y | x) for the given x and y."""
    target_full_instruct = Seq(
        text=MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target_instruct = Seq(
        text=instruct.text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )

    target_llm_no_losses = (
        target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=target_instruct,
            target=target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        .loss_batch.detach()
        .cpu()
    )

    target_llm_losses = (
        target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=target_full_instruct,
            target=target,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )
        .loss_batch.detach()
        .cpu()
    )

    return target_llm_no_losses - target_llm_losses


def select_and_evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    baseline,  # New here
    suffix,
    prompter,
    target_llm,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_losses,
    num_beams_in,
):
    num_chunks = cfg.train.q_params.num_chunks
    assert cfg.train.q_params.top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = cfg.train.q_params.top_k // (num_chunks * num_beams_in)
    all_next_token_candidate_ids = None

    for i in range(cfg.train.q_params.num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            cfg=cfg,
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            always_include_best=cfg.train.q_params.candidates.always_include_best
            and i == 0,
        )  # (ebs = bs * num_beams_in, num_samples_per_beam)

        candidate_beam_scores, candidate_losses = evaluate_next_token_candidates(
            cfg=cfg,
            instruct=instruct,
            target=target,
            baseline=baseline,
            suffix=suffix,
            prompter=prompter,
            target_llm=target_llm,
            next_token_candidate_ids=next_token_candidate_ids,
            next_dist_seq_basemodel=next_dist_seq_basemodel,
            next_dist_seq_prompter=next_dist_seq_prompter,
            prev_beam_scores=beam_scores,
            prev_losses=prev_losses,
        )  # (ebs, num_samples_per_beam)

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_losses = candidate_losses
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_losses = torch.cat(
                (candidate_losses, all_candidate_losses), dim=1
            )  # (ebs, i * num_samples_per_beam)
    return all_next_token_candidate_ids, all_candidate_beam_scores, all_candidate_losses


def evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    baseline,  # New here
    suffix,
    prompter,
    target_llm,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_losses,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )

    # extend to match the extended batch size
    instruct_rep = instruct.repeat_interleave(num_samples_per_beam, dim=0)
    target_rep = target.repeat_interleave(num_samples_per_beam, dim=0)
    baseline_rep = baseline.repeat_interleave(num_samples_per_beam, dim=0)
    if not suffix.is_empty:
        suffix_rep = suffix.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        suffix_rep = suffix

    full_suffix_rep = MergedSeq(seqs=[suffix_rep, q_sample_seq]).to_seq(
        merge_dtype="ids"
    )

    target_rewards = calculate_reward(
        cfg,
        prompter=prompter,
        target_llm=target_llm,
        instruct=instruct_rep,
        suffix=full_suffix_rep,
        target=target_rep,
    )
    baseline_rewards = calculate_reward(
        cfg,
        prompter=prompter,
        target_llm=target_llm,
        instruct=instruct_rep,
        suffix=full_suffix_rep,
        target=baseline_rep,
    )
    losses = target_rewards - baseline_rewards
    losses = torch.reshape(losses, (ebs, num_samples_per_beam))
    loss_delta = losses - prev_losses[:, None]  # (ebs, num_samples_per_beam)

    if cfg.train.q_params.selected_logprobs != 0:
        next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
        selected_logprobs_basemodel = torch.gather(
            next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
        )  # (ebs, num_samples_per_beam)
        beam_scores_delta = (
            cfg.train.q_params.selected_logprobs * selected_logprobs_basemodel.cpu()
            + loss_delta
        )
    else:
        beam_scores_delta = loss_delta
    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta

    return new_beam_scores, losses
