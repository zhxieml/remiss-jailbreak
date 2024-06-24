import argparse
import os

import pandas as pd

from src.utils import check_jailbroken, evaluate_gpt_batch, get_test_prefixes


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = pd.read_csv(args.input).to_dict(orient="list")
    test_prefixes = get_test_prefixes()

    gpt_responses = evaluate_gpt_batch(
        [p + s for p, s in zip(results["forbidden_prompt"], results["suffix"])],
        model=args.model,
        parallel=True,
    )
    _, gpt_jailbroken_list = check_jailbroken(
        seq=gpt_responses, test_prefixes=test_prefixes
    )
    results["gpt_response"] = gpt_responses
    results["gpt_jailbroken"] = gpt_jailbroken_list

    # save
    pd.DataFrame(results).to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="res/transfergpt_eval/tmp.csv")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    args = parser.parse_args()

    main(args)
