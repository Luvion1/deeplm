"""
BitNet b1.58 Ternary Quantization.

Implements absmean ternary quantization {-1, 0, +1} as described in:
"BitNet: Scaling 1-bit Transformers for Large Language Models" (Ma et al., 2023)
"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (Ma et al., 2024)

Uses straight-through estimator (STE) for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize(weight: torch.Tensor, scale: str = "absmean") -> tuple:
    """Quantize weight matrix to ternary {-1, 0, +1}.

    Args:
        weight: FP16/BF16/FP32 weight tensor
        scale: scaling method - "absmean" (default) or "absmax"

    Returns:
        (quantized_tensor, scale_factor) where quantized is in {-1, 0, +1}
    """
    if scale == "absmean":
        gamma = weight.abs().mean()
    elif scale == "absmax":
        gamma = weight.abs().max()
    else:
        raise ValueError(f"Unknown scale method: {scale}")

    if gamma < 1e-12:
        return torch.zeros_like(weight), torch.tensor(1.0, device=weight.device)

    # Ternary quantization: sign(weight) * clip(|weight|/gamma, 0, 1)
    # Simplified: {-1 if w > gamma, 0 if |w| <= gamma, +1 if w < -gamma}
    quantized = torch.zeros_like(weight)
    quantized[weight > gamma] = 1.0
    quantized[weight < -gamma] = -1.0

    return quantized, gamma


class TernaryQuantizeFn(torch.autograd.Function):
    """Straight-through estimator for ternary quantization."""

    @staticmethod
    def forward(ctx, weight, scale="absmean"):
        quantized, gamma = ternary_quantize(weight, scale)
        ctx.save_for_backward(weight)
        ctx.scale_method = scale
        return quantized * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients through as if identity
        return grad_output, None


def ternary_quantize_ste(weight: torch.Tensor, scale: str = "absmean") -> torch.Tensor:
    """Ternary quantization with straight-through estimator for training."""
    return TernaryQuantizeFn.apply(weight, scale)


class BitNetLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with BitNet b1.58 ternary quantization.

    During training: uses STE for gradient flow through quantization.
    During eval: returns dequantized ternary weights.
    """

    def __init__(self, in_features, out_features, bias=False, scale="absmean"):
        super().__init__(in_features, out_features, bias=bias)
        self.scale_method = scale

    def forward(self, x):
        if self.training:
            # Quantize weights with STE during training
            w_quantized = ternary_quantize_ste(self.weight, self.scale_method)
            return F.linear(x, w_quantized, self.bias)
        else:
            # Use ternary weights during inference
            w_quantized, gamma = ternary_quantize(self.weight, self.scale_method)
            return F.linear(x, w_quantized * gamma, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, bitnet=True, scale={self.scale_method}"


def apply_bitnet_quantization(model: nn.Module, scale: str = "absmean", verbose: bool = False) -> dict:
    """Apply BitNet b1.58 ternary quantization to all Linear layers in a model.

    Replaces nn.Linear with BitNetLinear for training, or quantizes weights in-place for inference.

    Args:
        model: PyTorch model
        scale: quantization scale method
        verbose: print quantization stats

    Returns:
        dict with quantization statistics
    """
    stats = {"total_linear": 0, "quantized": 0, "errors": 0, "params_before": 0, "params_after": 0}

    # Count params before
    for p in model.parameters():
        stats["params_before"] += p.numel()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats["total_linear"] += 1

            # Skip embedding-like layers and small layers
            if module.weight.shape[0] < 64 or module.weight.shape[1] < 64:
                if verbose:
                    print(f"  Skipping {name} (too small: {module.weight.shape})")
                continue

            try:
                # Quantize weights in-place (copy to avoid set_ on graph input)
                q_weight, gamma = ternary_quantize(module.weight, scale)
                module.weight.detach().copy_(q_weight * gamma)
                stats["quantized"] += 1

                if verbose:
                    # Compute quantization error
                    orig = module.weight.data
                    err = (orig - q_weight * gamma).abs().mean().item()
                    sparsity = (q_weight == 0).float().mean().item() * 100
                    print(f"  {name}: shape={list(module.weight.shape)}, "
                          f"scale={gamma:.4f}, err={err:.6f}, sparsity={sparsity:.1f}%")
            except Exception as e:
                stats["errors"] += 1
                if verbose:
                    print(f"  Error quantizing {name}: {e}")

    # Count params after (same, but stored as ternary)
    for p in model.parameters():
        stats["params_after"] += p.numel()

    stats["compression_ratio"] = stats["params_before"] / max(stats["params_after"], 1)
    return stats


def quantize_state_dict(state_dict: dict, scale: str = "absmean") -> dict:
    """Quantize a state dict to ternary weights.

    Returns a new state dict with ternary-quantized weights.
    """
    quantized = {}
    for key, tensor in state_dict.items():
        if tensor.dim() == 2 and tensor.shape[0] >= 64 and tensor.shape[1] >= 64:
            q_weight, gamma = ternary_quantize(tensor, scale)
            quantized[key] = q_weight * gamma
        else:
            quantized[key] = tensor
    return quantized
