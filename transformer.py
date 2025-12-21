import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, model_embedding_dim, max_len=5000, n=10000):
        """
        Initializes the PositionalEmbedding module. Positional encoding creates numerical representations in embedding space
        for each position in the input sequence. The number of positions is defined by the maximum sequence length the transformer
        will process. In the case of max_len = 5000 there will be generated 5000 positional embeddings. The positional embeddings are
        calculated with the following formulas: 
            For even positions in the sequence (e.g 0, 2, 4 etc): P(k, 2i) = sin(k/n^(2i/d)),
            For odd positions in the sequence (e.g 1, 3, 5 etc): P(k, 2i + 1) = sin(k/n^(2i/d)),
        where k is an position of an object in the input sequence, 0 < k < L/2 as we essentially split the sequence in two equal half which is processed by separate functions,
        d is the embedding dimensions of the input to the model, 
        n user defined parameter,
        i used for mapping column indices 0 < i < d/2, with a signle value of i maps to both sine and cosine functions.

        Args:
            model_embedding_dim (int): The embedding dimension of the model.
            max_len (int): The maximum sequence length the model will ever see.
            n (int): maximum wavelength of the positional encodings
        """
        super().__init__()

        # Create a tensor to store the positional embeddings
        # Shape: (max_len, d_model)
        self.positional_embeddings = torch.zeros(max_len, model_embedding_dim)

        # Create a tensor for the positions (0, 1, ..., max_len - 1)
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create the division term (the 1 / n^(2i / d_model) part)
        # Calculate it in log-space for better numerical stability
        div_term = torch.exp(torch.arange(0, model_embedding_dim, 2).float() * (-math.log(n) / model_embedding_dim))
        # Shape: (d_model / 2)

        # Apply sin to even indices (2i)
        self.positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (2i + 1)
        self.positional_embeddings[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so it can be added to batch data
        # New shape: (1, max_len, d_model)
        self.positional_embeddings = self.positional_embeddings.unsqueeze(0)

        # Register 'positional_embeddings' as a buffer. This makes it part of the model's
        # state, but not a parameter that gets updated by the optimizer.
        self.register_buffer('positional_embeddings', self.positional_embeddings)

    def forward(self, x):
        # Add the positional encodings to the input
        # We slice self.positional_embeddings to match the sequence length of x
        # self.positional_embeddings[:, :x.size(1), :] handles sequences shorter than max_len
        x = x + self.positional_embeddings[:, :x.size(1), :]
        return x
    

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_attention_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_attention_heads = n_attention_heads
        self.head_dim = embedding_dim // n_attention_heads # Dim per head (e.g., 512 / 8 = 64)

        assert (
            self.head_dim * n_attention_heads == self.embedding_dim
        ), "Embedding dimension must be divisible by number of heads"

        # Projects input to Q, K, V for all heads at once
        # Input: (batch, seq_len, embedding_dim) -> Output: (batch, seq_len, 3 * d_model)
        self.q_layer = nn.Linear(embedding_dim, embedding_dim)
        self.k_layer = nn.Linear(embedding_dim, embedding_dim)
        self.v_layer = nn.Linear(embedding_dim, embedding_dim)
        
        # Final linear layer (W_o) to combine head outputs
        # Input: (batch, seq_len, embedding_dim) -> Output: (batch, seq_len, d_model)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product(self, q, k, v, mask=None):
        """
        Calculates Attention(Q, K, V) = softmax((Q@K.T)/sqrt(d_k))@V
        Operates in parallel across all heads (n_attention_heads dim).
        q, k, v shape: (batch_size, n_attention_heads, sequence_length, head_dim)
        """
        d_k = q.size()[-1] # head_dim
        
        # Scores: (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) -> (batch, heads, seq_len, seq_len)
        scaled_dot_product = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scaled_dot_product, dim=-1)
        
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) -> (batch, heads, seq_len, head_dim)
        attention_scores = torch.matmul(attention, v)
        
        return attention_scores, attention

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        # Project Q, K, V
        # Shape: (batch, seq_len, embedding_dim)
        q = self.q_layer(query)
        k = self.k_layer(key)
        v = self.v_layer(value)

        # Reshape and permute for multi-head computation
        # Shape: (batch, seq_len, embedding_dim) -> (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch_size, q_len, self.n_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, k_len, self.n_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, v_len, self.n_attention_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention
        # values shape: (batch, n_attention_heads, q_len, head_dim)
        # attention shape: (batch, n_attention_heads, q_len, k_len)
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)

        # Combine/concatenate heads
        # Permute back: (batch, n_attention_heads, q_len, head_dim) -> (batch, q_len, n_attention_heads, head_dim)
        values = values.permute(0, 2, 1, 3)

        # Flatten heads: (batch, q_len, n_attention_heads, head_dim) -> (batch, q_len, embedding_dim)
        values = values.reshape(batch_size, q_len, self.embedding_dim)

        # Final projection
        # Shape: (batch, q_len, embedding_dim)
        attention_scores = self.output_layer(values)
        
        return attention_scores, attention
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_attention_heads, dropout_rate=0.1):
        super().__init__()

        self.multiheaded_self_attention = MultiheadedSelfAttention(embedding_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        attention_scores, attention_weights = self.multiheaded_self_attention(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + self.dropout1(attention_scores))
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout2(ff_output))

        return output, attention_weights


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_attention_heads, dropout_rate=0.1, use_cross_attention: bool = True):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention

        self.masked_multiheaded_self_attention = MultiheadedSelfAttention(embedding_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        if self.use_cross_attention:
            self.encoder_decoder_attention = MultiheadedSelfAttention(embedding_dim, n_attention_heads)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.norm2 = nn.LayerNorm(embedding_dim)
        else:
            self.encoder_decoder_attention = None
            self.dropout2 = None
            self.norm2 = None

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, x, target_mask, encoder_output=None, encoder_mask=None):
        """
        x:               Target sequence embeddings
        target_mask:     Look-ahead mask for target sequence (for masked self-attention)
        encoder_output:  Output from the final encoder block (K, V for cross-attention). Optional.
        encoder_mask:    Padding mask for source sequence (for cross-attention). Optional.
        """

        masked_attn_output, self_attn_weights = self.masked_multiheaded_self_attention(
            query=x, key=x, value=x, mask=target_mask
        )
        x = self.norm1(x + self.dropout1(masked_attn_output))

        cross_attn_weights = None
        
        if self.use_cross_attention:
            if encoder_output is None:
                raise ValueError("encoder_output must be provided when use_cross_attention is True")
                
            cross_attn_output, cross_attn_weights = self.encoder_decoder_attention(
                query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
            )
            x = self.norm2(x + self.dropout2(cross_attn_output))

        ff_output = self.feed_forward(x)
        output = self.norm3(x + self.dropout3(ff_output))
        
        return output, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 target_vocab_size: int, 
                 model_embedding_dim: int, 
                 n_encoder_blocks: int, 
                 n_decoder_blocks: int, 
                 n_attention_heads: int, 
                 max_len: int = 5000, 
                 dropout_rate: float = 0.1,
                 use_cross_attention: bool = True):
        """
        Args:
            src_vocab_size (int): Size of the source vocabulary.
            target_vocab_size (int): Size of the target vocabulary.
            model_embedding_dim (int): Dimension of embeddings (d_model).
            n_encoder_blocks (int): Number of encoder blocks.
            n_decoder_blocks (int): Number of decoder blocks.
            n_attention_heads (int): Number of attention heads.
            max_len (int): Maximum sequence length for positional encoding.
            dropout_rate (float): Dropout probability.
            use_cross_attention (bool): If True, builds a full Encoder-Decoder model.
                                        If False, builds a Decoder-Only model 
                                        (src_vocab_size and n_encoder_blocks are ignored).
        """
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        if self.use_cross_attention:
            self.src_embedding = nn.Embedding(src_vocab_size, model_embedding_dim)
            self.encoder_blocks = nn.ModuleList(
                [TransformerEncoderBlock(model_embedding_dim, n_attention_heads, dropout_rate) 
                 for _ in range(n_encoder_blocks)]
            )
        else:
            self.src_embedding = None
            self.encoder_blocks = None
            
        self.target_embedding = nn.Embedding(target_vocab_size, model_embedding_dim)
        self.pos_embed = PositionalEmbedding(model_embedding_dim, max_len)
        
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(
                model_embedding_dim, 
                n_attention_heads, 
                dropout_rate, 
                use_cross_attention=self.use_cross_attention
             ) 
             for _ in range(n_decoder_blocks)]
        )
        
        self.output_layer = nn.Linear(model_embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, src, src_mask):
        if not self.use_cross_attention:
            return None
            
        src_embedded = self.dropout(self.pos_embed(self.src_embedding(src)))
        
        encoder_output = src_embedded
        for block in self.encoder_blocks:
            encoder_output, _ = block(encoder_output, src_mask)
        return encoder_output

    def decode(self, target, encoder_output, target_mask, src_mask):
        target_embedded = self.dropout(self.pos_embed(self.target_embedding(target)))
        
        decoder_output = target_embedded
        for block in self.decoder_blocks:
            decoder_output, _, _ = block(decoder_output, target_mask, encoder_output, src_mask)
        return decoder_output

    def forward(self, src, target, src_mask, target_mask):
        if self.use_cross_attention:
            encoder_output = self.encode(src, src_mask)
        else:
            encoder_output = None
        
        decoder_output = self.decode(target, encoder_output, target_mask, src_mask)
        output = self.output_layer(decoder_output)
        
        return output

    def create_target_mask(self, target):
        _, target_len = target.shape
        mask = torch.triu(torch.ones(target_len, target_len, device=target.device), diagonal=1).bool()
        return ~mask.unsqueeze(0).unsqueeze(1) 

    def create_padding_mask(self, seq, pad_token_idx=0):
        mask = (seq != pad_token_idx).unsqueeze(1).unsqueeze(2)
        return mask