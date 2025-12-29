import argparse

import torch
import torch.nn as nn


from transformer import Transformer



def main(inference_configs):
    loaded_statedict = None
    model = Transformer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Configurations for running inference")
    parser.add_argument(
        "--statedict_path"
    )
    inference_configs = parser.parse_args()
    main(inference_configs)