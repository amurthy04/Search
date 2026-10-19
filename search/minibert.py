"""Minimal traceable BERT encoder (loads HF weights) for CoreML / ANE export. CLS or mean pooling, L2-normalized."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, h, heads, inter, eps):
        super().__init__()
        self.h, self.heads = h, heads
        self.d = h // heads
        self.scale = 1.0 / math.sqrt(self.d)
        self.q = nn.Linear(h, h); self.k = nn.Linear(h, h); self.v = nn.Linear(h, h); self.o = nn.Linear(h, h)
        self.ln1 = nn.LayerNorm(h, eps=eps)
        self.ff1 = nn.Linear(h, inter); self.ff2 = nn.Linear(inter, h)
        self.ln2 = nn.LayerNorm(h, eps=eps)

    def forward(self, x, bias, B, S):
        q = self.q(x).view(B, S, self.heads, self.d).transpose(1, 2)
        k = self.k(x).view(B, S, self.heads, self.d).transpose(1, 2)
        v = self.v(x).view(B, S, self.heads, self.d).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale + bias
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, S, self.h)
        x = self.ln1(x + self.o(y))
        x = self.ln2(x + self.ff2(F.gelu(self.ff1(x))))
        return x


class MiniBert(nn.Module):
    def __init__(self, cfg, pooling="cls", fixed=None):
        super().__init__()
        self.fixed = fixed  # (B, S) constants for tracing fixed-shape graphs (CoreML)
        h = cfg.hidden_size
        self.word = nn.Embedding(cfg.vocab_size, h)
        self.pos = nn.Embedding(cfg.max_position_embeddings, h)
        self.tok_type = nn.Embedding(cfg.type_vocab_size, h)
        self.ln = nn.LayerNorm(h, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList([Layer(h, cfg.num_attention_heads, cfg.intermediate_size, cfg.layer_norm_eps) for _ in range(cfg.num_hidden_layers)])
        self.pooling = pooling
        self.register_buffer("pos_ids", torch.arange(cfg.max_position_embeddings), persistent=False)

    def forward(self, input_ids, attention_mask):
        B, S = self.fixed if self.fixed else input_ids.shape
        x = self.ln(self.word(input_ids) + self.pos(self.pos_ids[:S]).unsqueeze(0) + self.tok_type.weight[0])
        bias = (1.0 - attention_mask.to(x.dtype)).unsqueeze(1).unsqueeze(2) * -1e4
        for l in self.layers:
            x = l(x, bias, B, S)
        if self.pooling == "cls":
            e = x[:, 0, :]
        else:
            m = attention_mask.to(x.dtype).unsqueeze(-1)
            e = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return e / torch.linalg.norm(e, dim=1, keepdim=True)

    @staticmethod
    def from_hf(name, pooling="cls", fixed=None):
        from transformers import AutoModel
        hf = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        sd = hf.state_dict()
        m = MiniBert(hf.config, pooling, fixed)
        pre = "" if "embeddings.word_embeddings.weight" in sd else "bert."
        g = lambda k: sd[pre + k]
        m.word.weight.data = g("embeddings.word_embeddings.weight"); m.pos.weight.data = g("embeddings.position_embeddings.weight")
        m.tok_type.weight.data = g("embeddings.token_type_embeddings.weight")
        m.ln.weight.data = g("embeddings.LayerNorm.weight"); m.ln.bias.data = g("embeddings.LayerNorm.bias")
        for i, l in enumerate(m.layers):
            p = f"encoder.layer.{i}."
            for mod, key in [(l.q, "attention.self.query"), (l.k, "attention.self.key"), (l.v, "attention.self.value"),
                             (l.o, "attention.output.dense"), (l.ff1, "intermediate.dense"), (l.ff2, "output.dense")]:
                mod.weight.data = g(p + key + ".weight"); mod.bias.data = g(p + key + ".bias")
            l.ln1.weight.data = g(p + "attention.output.LayerNorm.weight"); l.ln1.bias.data = g(p + "attention.output.LayerNorm.bias")
            l.ln2.weight.data = g(p + "output.LayerNorm.weight"); l.ln2.bias.data = g(p + "output.LayerNorm.bias")
        return m.eval()
