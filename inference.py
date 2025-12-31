import argparse

import torch
import torch.nn as nn


from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(inference_configs):
    loaded_configs = torch.load(inference_configs.statedict_path, weights_only=False)
    model = Transformer(1, loaded_configs['tokenizer'].vocab_size, loaded_configs['configs'].embed_dim, loaded_configs['configs'].n_encoder_blocks, loaded_configs['configs'].n_decoder_blocks, loaded_configs['configs'].n_attention_heads, loaded_configs['configs'].context_len, use_cross_attention=False).to(device)
    model.load_state_dict(loaded_configs['model_state_dict'])

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    res = ''.join(loaded_configs['tokenizer'].decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Configurations for running inference")
    parser.add_argument(
        "--statedict_path"
    )
    inference_configs = parser.parse_args()
    main(inference_configs)