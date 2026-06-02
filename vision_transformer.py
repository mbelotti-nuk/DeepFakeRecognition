import torch
import torch.nn as nn
from torch.nn.functional import softmax

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, class_size, n_layers, n_heads, embed_size, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_size = embed_size

        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_size, kernel_size=patch_size, stride=patch_size)

        # 2. CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        
        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_size))
        self.pos_drop = nn.Dropout(p=dropout)

        self.encoder = Encoder(n_layers, n_heads, embed_size, dropout)
        
        self.projection = nn.Linear(embed_size, class_size)
        self.register_buffer('temperature', 0.2 * torch.ones(class_size))
        self.init_weights()

    def forward(self, x):

        B = x.shape[0]

        x = self.patch_embed(x) # [B, C, H, W] -> [B, embed_size, H_patch, W_patch]
        x = x.reshape(B, self.embed_size, -1) # [B, embed_size, num_patches]
        x = x.permute(0, 2, 1).contiguous() # [B, num_patches, embed_size]

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, embed_size]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)

        # Extract class token output
        cls_output = x[:, 0] # [B, embed_size]
        
        return cls_output
    
    def get_logits(self, x):
        return self.projection(x)
    
    def get_probabilities(self, x):
        logits = self.get_logits(x)
        return softmax(logits / self.temperature.to(x.device), dim=-1)
    
    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, n_embed, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, n_embed, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_embed) 

    def forward(self, x):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, n_embed, dropout=0.1):
        super().__init__()
        self.multi_head_attn = MultiHeadAttentionWrapper(n_heads, n_embed)
        self.norm_1 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed)
        self.norm_2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.norm_1(x)
        attn_out = self.multi_head_attn(x,x,x)
        x = x + self.dropout(attn_out)
        x = self.norm_2(x)
        x = x + self.dropout(self.ff(x))
        return x

class MultiHeadAttentionWrapper(nn.Module):
    """
        Multi head attention in ones step
    """
    def __init__(self, n_heads, n_embed):
        super().__init__()
        assert n_embed % n_heads == 0, f"The embedding size (={n_embed}) and the number of heads (={n_heads}) must be divisible"
        
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads

        self.K = nn.Linear(n_embed, n_embed, bias=False)
        self.Q = nn.Linear(n_embed, n_embed, bias=False)
        self.V = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, query, key, value):
        B, Tq, Cq = query.shape 
        B, Tk, Ck = key.shape

        key   = self.K(key)
        query = self.Q(query)
        value = self.V(value)

        key   = key.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)
        query = query.view(B, Tq, self.n_heads, self.head_size).transpose(1, 2)
        value = value.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)

        affinities = self.attn_weights(query, key)
        attn = affinities @ value                        # (B, N_HEADS, Tq, HEAD_SIZE)
        attn = attn.transpose(1, 2).reshape(B, Tq, Cq) 
        return attn


    def attn_weights(self, query, key)->torch.Tensor:
        # compute attention scores
        affinities = query @ key.transpose(-2,-1) # (B, N_HEADS, Tq, HEAD_SIZE) @ (B, N_HEADS, HEAD_SIZE, Tk) = (B, N_HEADS,Tq,Tk)
        # scale attention
        head_size = query.size(-1)
        affinities = affinities * head_size ** (-0.5)   
        # compute probabilities
        return affinities.softmax(-1) 

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_embed * 4, n_embed)
        )
    def forward(self, x):
        return self.linear(x)