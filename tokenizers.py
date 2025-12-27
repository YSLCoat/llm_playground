import torch

class CharacterLevelTokenizer():
    def __init__(self, text: str, vocab: dict = None):
        self.text = text
        
        if vocab is None:
            unique_chars = sorted(list(set(self.text)))
            
            self.character_to_integer = { ch: i for i, ch in enumerate(unique_chars) }
            self.integer_to_character = { i: ch for i, ch in enumerate(unique_chars) }
            
            self.unk_token = "<UNK>"
            if self.unk_token not in self.character_to_integer:
                unk_idx = len(self.character_to_integer)
                self.character_to_integer[self.unk_token] = unk_idx
                self.integer_to_character[unk_idx] = self.unk_token
                
        else:
            self.character_to_integer = vocab
            self.integer_to_character = {i: ch for ch, i in vocab.items()}
            self.unk_token = "<UNK>"

        self.vocab_size = len(self.character_to_integer)
        self.unk_index = self.character_to_integer.get(self.unk_token)

        self.tokens = torch.tensor(
            [self.character_to_integer.get(c, self.unk_index) for c in self.text], 
            dtype=torch.long
        )

    def decode(self, tokens: list[int]) -> list[str]:
        return [self.integer_to_character.get(token, self.unk_token) for token in tokens]