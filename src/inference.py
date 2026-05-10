import argparse
import sys


TARGETS = {
    "model-a": "model_a_inference",
    "model-a-generator": "model_a_generation",
    "model-b": "model_b",
}


def main():
    parser = argparse.ArgumentParser(description="Rubric entrypoint for model inference.")
    parser.add_argument("--target", choices=TARGETS, default="model-a")
    args, remaining = parser.parse_known_args()

    module_name = TARGETS[args.target]
    module = __import__(module_name)
    sys.argv = [f"{module_name}.py", *remaining]
    module.main()


if __name__ == "__main__":
    main()
