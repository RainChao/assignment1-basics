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


class RoPE(nn.Module):
    """
    旋转位置编码
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化RoPE
        Args:
            theta: 旋转角度基数
            d_k: 输入Q或K向量的维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算旋转复数矩阵
        self.register_buffer("rope", self._precompute_freqs_cis(), persistent=False)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        """
        预计算频率和相位
        Returns:
            形状为(max_seq_len, d_k)的张量，包含旋转位置编码
        """
        # 计算\theta_i序列，也就是频率序列
        # theta_i = 1 / { theta^{2i / d_k} }
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[:(self.d_k // 2)] / self.d_k))
        # 生成序列索引m [0, 1, ..., max_seq_len-1]
        seq_idx = torch.arange(0, self.max_seq_len, device=self.device)
        # 计算 m * \theta_i 矩阵
        freqs = einsum(seq_idx, freqs, "seq, d -> seq d")

        # 复数化
        # freqs[m][i] = m * \theta_i
        # freqs_cis[m][i] = 1 * e^{i * m * \theta_i}
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(..., seq_len, d_k)
            token_positions: 位置索引，形状为(..., seq_len)
        Returns:
            旋转位置编码后的张量，形状为(..., seq_len, d_k)
        """
        # 将维度分组
        x_ = rearrange(x, "... seq (d two) -> ... seq d two", two=2).float()
        # 转为复数(... seq (d 2) )
        x_ = torch.view_as_complex(x_)

        # 根据token_positions获取对应的位置的频率
        rope_pos = self.rope[token_positions]  # (batch, ..., seq_len, d_k // 2)

        # 旋转，之后转回实数域并展平
        x_out = rearrange(torch.view_as_real(x_ * rope_pos), "... seq d two -> ... seq (d two)", two=2)
        
        return x_out.to(x.dtype)  # 转回原始dtype


class MultiheadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """
    带有旋转位置编码的多头自注意力
    """
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
            token_positions: 位置索引，形状为(batch, ..., seq_len)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        QKV = self.w_qkv(x)  # (batch, ..., seq_len, head * d_k * 3)
        # 分割Q、K、V
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three=3, head=self.num_heads)

        # 对Q，K使用RoPE，head视为batch维度
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # 因果掩码：(seq_len_q, seq_len_k)
        # 位置i的query不能分配注意力给位置j的key（j>i）
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)  # (batch, ..., head, seq_len, d_k)
        
        # 将多头拼接回去
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        
        return self.w_o(atten)  # (batch, ..., seq_len, d_model)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight.to(in_dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int, theta: float=10000.0, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.self_attention = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device)
        self.feed_forward = SWiGLUFeedForward(d_model, d_ff, device=device)
        self.layer_norm1 = RMSNorm(d_model, device=device)
        self.layer_norm2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], dtype=torch.int, device=x.device)
        atten_output = self.self_attention(self.layer_norm1(x), token_positions)
        x2 = x + atten_output
        ffn_output = self.feed_forward(self.layer_norm2(x2))
        return x2 + ffn_output


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float=10000.0,
        device=None):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device)
        self.tf_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, context_length, d_ff, theta, device=device) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device)
        self.output_embedding = Linear(d_model, vocab_size, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for tf_block in self.tf_blocks:
            x = tf_block(x)
        x = self.ln_final(x)
        return self.output_embedding(x)
