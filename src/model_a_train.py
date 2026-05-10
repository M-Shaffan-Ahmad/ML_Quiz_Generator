import argparse
import sys


TARGETS = {
    "classical": ("run_model_a_evals", "Train/evaluate the final classical LR + XGBoost Model A."),
    "generator": ("train_model_a_generator", "Train the supervised question generator."),
    "unsupervised": ("train_model_a_unsupervised", "Train the unsupervised/semi-supervised clustering artifacts."),
    "bert": ("train_bert_model_a", "Fine-tune the optional answer-text BERT backend."),
    "bert-mc": ("train_bert_multiple_choice_model_a", "Fine-tune the BERT multiple-choice backend."),
}


def main():
    parser = argparse.ArgumentParser(description="Rubric entrypoint for Model A training.")
    parser.add_argument("--target", choices=TARGETS, default="classical")
    args, remaining = parser.parse_known_args()

    module_name = TARGETS[args.target][0]
    module = __import__(module_name)
    sys.argv = [f"{module_name}.py", *remaining]
    module.main()


if __name__ == "__main__":
    main()
