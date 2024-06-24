target_llm=${1:-"llama2_chat"}
target_device=${2:-"cuda:0"}
prompter_device=${3:-"cuda:1"}
lambda=${4:-150}
logprobs=${5:-2}
opt_type=${6:-"remiss"}
base_dir=${7:-"res"}

jobname="target${target_llm}_lambda${lambda}_logprobs${logprobs}_opt${opt_type}_$(date +%s)"
output_dir=${base_dir}/${jobname}
mkdir -p ${output_dir}

python main.py \
    --config-name=eval \
    target_llm=${ori_target_llm} \
    target_llm.llm_params.device=${target_device} \
    prompter=${prompter} \
    prompter.llm_params.device=${prompter_device} \
    prompter.llm_params.lora_params.warmstart=true \
    prompter.llm_params.lora_params.lora_checkpoint=${ckpt} \
    wandb_params.enable_wandb=false \
    eval.batch_size=52 | | tee ${output_dir}/eval.log
