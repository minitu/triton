import torch

batch, sentence_length, embedding_dim = 20, 5, 10
layer_norm = torch.nn.LayerNorm(embedding_dim)

@torch.compile
def ln_fp8(
    x: torch.Tensor
) -> torch.Tensor:
    out = layer_norm(x)
    out = out.to(torch.float8_e4m3fn) # Doesn't work with TorchInductor
    return out

embedding = torch.randn(batch, sentence_length, embedding_dim)
output = ln_fp8(embedding)
print(output)
