target_llm=${1:-"llama2_chat"}
target_device=${2:-"cuda:0"}
prompter_device=${3:-"cuda:1"}
lambda=${4:-150}
logprobs=${5:-2}
opt_type=${6:-"remiss"}
base_dir=${7:-"res/search"}

jobname="target${target_llm}_lambda${lambda}_logprobs${logprobs}_opt${opt_type}_$(date +%s)"
output_dir=${base_dir}/${jobname}
mkdir -p ${output_dir}


python eval_search.py \
    --config-name=train \
    verbose=true \
    train.dataset_pth=data/subset.csv \
    target_llm=${target_llm} \
    train.q_params.num_beams=4 \
    train.q_params.num_chunks=1 \
    train.batch_size=16 \
    target_llm.llm_params.device=${target_device} \
    prompter.llm_params.device=${prompter_device} \
    train.q_params.lambda_val=${lambda} \
    train.q_params.selected_logprobs=${logprobs} \
    wandb_params.id=${jobname} \
    train.opt_type=${opt_type} | tee ${output_dir}/search.log
