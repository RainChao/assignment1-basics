import torch
import torch.nn as nn
from math import sqrt
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype) / sqrt(in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "b ... d_in, d_out d_in -> b ... d_out")


class SWiGLUFeedForward(nn.Module):
    """
    使用 SWiGLU 激活函数的前馈神经网络
    """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        初始化
        Args:
            d_model: 输入特征的维度
            d_ff: SWiGLU的中间层维度，如果为None，则等于8/3 * d_model（舍入到最接近64的值）
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(8 / 3 * d_model)
            self.d_ff = (self.d_ff + 63) // 64 * 64
        else:
            self.d_ff = d_ff
        self.weight1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 W2(SiLU(W1 x) \odot (W3 x))
        Args:
            x: 输入张量，形状为(batch, ..., d_model)
        Returns:
            输出张量，形状为(batch, ..., d_model)
        """
        # 计算SWiGLU
        w1_x = self.weight1(x)
        w3_x = self.weight3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        swiglu = silu * w3_x
        return self.weight2(swiglu)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None):
    """
    计算缩放点积注意力
    Args:
        query: 查询张量，形状为(batch, num_heads, seq_len, d_k)
        key: 键张量，形状为(batch, num_heads, seq_len, d_k)
        value: 值张量，形状为(batch, num_heads, seq_len, d_v)
        mask: 掩码张量，形状为(batch, num_heads, seq_len, seq_len)
    Returns:
        输出张量，形状为(batch, num_heads, seq_len, d_v)
    """
    score = einsum(query, key, "b ... i d_k, b ... j d_k -> b ... i j") / sqrt(query.shape[-1])
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(score, dim=-1)
    return einsum(attention, value, "b ... i j, b ... j d_v -> b ... i d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        self.w_qkv = Linear(d_model, 3 * d_model)
        self.w_o = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        QKV = self.w_qkv(x)  # (batch, ..., seq_len, head * d_k * 3)
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(x.device)
        atten = scaled_dot_product_attention(Q, K, V, mask)
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        return self.w_o(atten)
