target_device=${1:-"cuda:0"}
prompter_device=${2:-"cuda:1"}
base_dir=${3:-"res/detect_gt"}
mkdir -p $base_dir

for model_idx in 1 2 3 4 5; do
    target_llm=poisoned_llama2_t${model_idx}

    for trigger_idx in 1 2 3 4 5; do
        echo "Running $target_llm on trigger $trigger_idx"
        jobname=model${target_llm}_trigger${trigger_idx}
        output_dir=${base_dir}/${jobname}

        if [ -f ${output_dir} ]; then
            echo "Skipping ${jobname}"
            continue
        fi

        python backdoor.py \
            --config-name=detect \
            target_llm=${target_llm} \
            target_llm.llm_params.device=${target_device} \
            prompter=llama2_detect \
            prompter.llm_params.device=${prompter_device} \
            wandb_params.enable_wandb=false \
            detect.batch_size=50 \
            detect.trigger=${trigger_idx} \
            detect.gt_trigger=${model_idx} \
            output_dir=${output_dir} \
            detect.save_path=${output_dir}/result.jsonl
    done
done
