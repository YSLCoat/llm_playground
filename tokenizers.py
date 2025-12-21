import torch

class CharacterLevelTokenizer():
    def __init__(self, text: list[str]):
        self.text = text
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.character_to_integer = { character:integer for integer,character in enumerate(self.chars) }
        self.integer_to_character = { integer:character for integer,character in enumerate(self.chars) }

        self.tokens = torch.tensor([self.character_to_integer[character] for character in text], dtype=torch.long)

    def decode(self, tokens: list[int]) -> list[str]:
        text = [self.integer_to_character[token] for token in tokens]
        return text
    
    