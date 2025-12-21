import argparse 
import pathlib

def parse_input():
    parser = argparse.ArgumentParser(description="PyTorch LLM Playground")

    parser.add_argument(
        "--data-dir",
        "data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to rootdir containing data"
    )

    configs = parser.parse_args()
    return configs