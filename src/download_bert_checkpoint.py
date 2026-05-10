import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download and save a local BERT sequence-classification checkpoint."
    )
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="models/model_a/neural/bert")
    parser.add_argument("--num-labels", type=int, default=2)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Run `../venv/bin/pip install -r bts/requirements-bert.txt` first."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved local BERT checkpoint to: {output_dir.resolve()}")
    print("Note: this is a pretrained checkpoint with a newly initialized classification head.")
    print("Fine-tune it with train_bert_model_a.py before using it for final evaluation.")


if __name__ == "__main__":
    main()
