"""
Fragmento ilustrativo: Scaled Dot-Product Attention implementado a mano
en PyTorch, sin usar nn.MultiheadAttention, para ver la fórmula traducida
directamente a código.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Requisitos sugeridos:
    pip install torch
"""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                  mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q, K, V: tensores de forma (batch, n_tokens, d_k)
    mask: tensor opcional de forma (n_tokens, n_tokens) para atención causal
          (relleno con -inf en las posiciones futuras).
    """
    d_k = Q.size(-1)

    # 1. Q K^T -> compatibilidad entre cada par de tokens
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, n_tokens, n_tokens)

    # 2. Escalar por sqrt(d_k) para estabilizar el softmax
    scores = scores / math.sqrt(d_k)

    # 3. (Opcional) máscara causal: impedir ver tokens futuros
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4. softmax -> pesos de atención (suman 1 por fila)
    attention_weights = F.softmax(scores, dim=-1)

    # 5. Promedio ponderado de los Values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class SimpleMultiHeadAttention(torch.nn.Module):
    """Versión simplificada de multi-head attention, con fines didácticos."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible entre n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch_size, n_tokens, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Separar en n_heads cabezas: (batch, n_heads, n_tokens, d_k)
        def split_heads(t):
            return t.view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Volver a juntar las cabezas: (batch, n_tokens, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tokens, d_model)

        return self.W_o(out), attn_weights


if __name__ == "__main__":
    batch, n_tokens, d_model, n_heads = 1, 5, 16, 4

    x = torch.randn(batch, n_tokens, d_model)

    # Máscara causal: token i solo puede ver tokens <= i
    causal_mask = torch.tril(torch.ones(n_tokens, n_tokens))

    mha = SimpleMultiHeadAttention(d_model, n_heads)
    output, attn_weights = mha(x, mask=causal_mask)

    print("Forma de salida:", output.shape)          # (1, 5, 16)
    print("Forma de pesos de atención:", attn_weights.shape)  # (1, 4, 5, 5)
