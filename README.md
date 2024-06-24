# Jailbreaking as a Reward Misspecification Problem

This repository is the official implementation of our paper [Jailbreaking as a Reward Misspecification Problem](https://arxiv.org/abs/2406.14393).

## Overview

Aligned LLMs are highly vulnerable to adversarial attacks due to *reward misspecification*—when the alignment process fails to fully capture human judgment nuances, allowing the model to exploit these errors. With our system, ReMiss, we identify and generate adversarial prompts that expose these flaws, achieving high attack success rates while ensuring the prompts remain human-readable. This approach helps us better understand and improve the robustness of aligned LLMs.

![alt text](assets/image.png)


## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Training

To train ReMiss targeting `Llama-2-7b-chat`, run this command:

```
bash scripts/train.sh llama2_chat
```

You can alter the argument (e.g., `vicuna_chat`) to attack other models.
Here is an [example](data/vicuna_chat_response.csv) of the generated adversarial suffixes.

## Evaluation & More Analysis

### LlamaGuard Evaluation
By default, evaluation results on the test set are saved to `${OUTPUT_DIR}/suffix_dataset`.
You can also evaluate the responses with [LlamaGuard](https://huggingface.co/meta-llama/LlamaGuard-7b):

```
python eval_llamaguard.py --input response.csv --output response_llamaguard.csv
```

### Backdoor Detection Analysis
You can also reproduce our results on backdoor detection and transfer attacks.
For backdoor detection, run the script `scripts/backdoor.sh` and then visualize with `notebooks/backdoor.ipynb`.

### Transfer Attacks
To transfer adversarial attacks to `gpt-3.5-turbo-0301`, you can use the following command:

```
export OPENAI_API_KEY=${YOUR_API_KEY}
python eval_transfergpt.py --input response.csv --output response_gpt.csv --model gpt-3.5-turbo-0301
```

## Citation

If you find this useful in your research, please consider citing:

```
@misc{xie2024jailbreaking,
      title={Jailbreaking as a Reward Misspecification Problem},
      author={Zhihui Xie and Jiahui Gao and Lei Li and Zhenguo Li and Qi Liu and Lingpeng Kong},
      year={2024},
      eprint={2406.14393},
      archivePrefix={arXiv}
}
```

## Acknowledgement

Our code is based on the excellent work [AdvPrompter](https://github.com/facebookresearch/advprompter) ([license](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)).
We also thank Fan Zhou, Xiachong Feng, and Sheng Wang for their valuable feedback.