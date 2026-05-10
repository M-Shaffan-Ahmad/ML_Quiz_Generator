import argparse
import sys


TARGETS = {
    "model-a-classical": "compare_model_a_supervised",
    "model-a-bert": "evaluate_bert_multiple_choice_model_a",
    "model-a-bert-text": "evaluate_bert_model_a",
    "model-b": "train_model_b",
    "model-b-text": "evaluate_model_b_generation",
}


def main():
    parser = argparse.ArgumentParser(description="Rubric entrypoint for evaluation scripts.")
    parser.add_argument("--target", choices=TARGETS, default="model-a-classical")
    args, remaining = parser.parse_known_args()

    module_name = TARGETS[args.target]
    module = __import__(module_name)
    sys.argv = [f"{module_name}.py", *remaining]
    module.main()


if __name__ == "__main__":
    main()
