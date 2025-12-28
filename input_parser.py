import argparse 
import pathlib

def parse_input_to_configs():
    parser = argparse.ArgumentParser(description="PyTorch LLM Playground")

    parser.add_argument(
        "-d",
        "--datadir",
        type=pathlib.Path,
        required=True,
        help="Path to rootdir containing data"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs"
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=200,
        help="Context length of tokens"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="Model embedding dim"
    )
    parser.add_argument(
        "--n_encoder_blocks",
        type=int,
        default=0,
        help="Number of encoder blocks for transformer"
    )
    parser.add_argument(
        "--n_decoder_blocks",
        type=int,
        default=6,
        help="Number of decoder blocks"
    )
    parser.add_argument(
        "--n_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )

    configs = parser.parse_args()
    return configs