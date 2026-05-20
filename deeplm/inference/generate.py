"""
Inference utilities for Deeplm.

Supports:
- Model loading from checkpoint
- Text generation with KV cache
- Chat-style multi-turn conversation
- Thinking mode (optional)
"""
import torch
import json
import os
from typing import Optional, List, Dict

from ..config import DeeplmConfig
from ..model.deeplm import DeeplmModel


def load_model_for_inference(model_path: str, config_path: Optional[str] = None,
                             device: str = None) -> tuple:
    """Load Deeplm model for inference.

    Args:
        model_path: Path to directory containing model.pt and config.json
        config_path: Optional path to config.json (default: model_path/config.json)
        device: Device to load model on (default: cuda if available)

    Returns:
        (model, config) tuple
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config_path is None:
        config_path = os.path.join(model_path, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DeeplmConfig(
            vocab_size=config_dict.get("vocab_size", 128000),
        )
        config.architecture.hidden_size = config_dict.get("hidden_size", 384)
        config.architecture.num_layers = config_dict.get("num_layers", 8)
    else:
        config = DeeplmConfig()

    model = DeeplmModel(config)

    model_pt = os.path.join(model_path, "model.pt")
    if os.path.exists(model_pt):
        sd = torch.load(model_pt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded model from {model_pt}")
    else:
        print(f"No checkpoint found at {model_pt}, using random weights")

    model.to(device)
    model.eval()
    return model, config


def generate_text(model: DeeplmModel, config: DeeplmConfig, prompt: str,
                  tokenizer=None, max_new_tokens: int = 1024,
                  temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                  do_sample: bool = True, repetition_penalty: float = 1.05,
                  thinking_mode: bool = False) -> str:
    """Generate text from a prompt.

    Args:
        model: Loaded DeeplmModel
        config: Model config
        prompt: Input text
        tokenizer: Tokenizer (HuggingFace or tokenizers.Tokenizer)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        do_sample: Whether to sample or use greedy
        repetition_penalty: Penalty for repeated tokens
        thinking_mode: Enable thinking mode (wrap output in think tags)

    Returns:
        Generated text
    """
    device = next(model.parameters()).device

    # Tokenize input
    if tokenizer is not None:
        input_ids = _tokenize(tokenizer, prompt, device)
        eos_token_id = _get_eos_token_id(tokenizer)
    else:
        input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        eos_token_id = None

    # Generate
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    # Decode output
    if tokenizer is not None:
        output_text = _decode(tokenizer, generated[0])
    else:
        output_text = str(generated[0].tolist())

    # Handle thinking mode
    if thinking_mode:
        output_text = _extract_thinking(output_text)

    return output_text


def chat(model: DeeplmModel, config: DeeplmConfig, messages: List[Dict[str, str]],
         tokenizer=None, max_new_tokens: int = 1024,
         temperature: float = 0.7, thinking_mode: bool = False) -> str:
    """Multi-turn chat with Deeplm.

    Args:
        model: Loaded DeeplmModel
        config: Model config
        messages: List of {role, content} dicts
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        thinking_mode: Enable thinking mode

    Returns:
        Assistant response
    """
    prompt = _format_messages(messages)
    return generate_text(
        model, config, prompt, tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
    )


def _tokenize(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Tokenize text using either HuggingFace or tokenizers.Tokenizer."""
    # Check if it's a HuggingFace tokenizer
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "return_tensors"):
        return tokenizer.encode(text, return_tensors="pt").to(device)

    # tokenizers.Tokenizer
    if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "return_tensors"):
        enc = tokenizer.encode(text)
        return torch.tensor([enc.ids], dtype=torch.long, device=device)

    raise ValueError(f"Unknown tokenizer type: {type(tokenizer)}")


def _decode(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode token IDs using either HuggingFace or tokenizers.Tokenizer."""
    ids = token_ids.tolist()

    # HuggingFace tokenizer
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=True)

    # tokenizers.Tokenizer
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)

    return str(ids)


def _get_eos_token_id(tokenizer) -> Optional[int]:
    """Get EOS token ID from tokenizer."""
    if hasattr(tokenizer, "eos_token_id"):
        return tokenizer.eos_token_id
    if hasattr(tokenizer, "token_to_id"):
        return tokenizer.token_to_id("<|end_of_sentence|>")
    return None


def _format_messages(messages: List[Dict[str, str]]) -> str:
    """Format conversation messages into a prompt."""
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"<|system|>{content}\n"
        elif role == "user":
            prompt += f"<|user|>{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>{content}\n"

    prompt += "<|assistant|>"
    return prompt


def _extract_thinking(text: str) -> str:
    """Extract thinking content from output with think tags."""
    think_start = "<|think_start|>"
    think_end = "<|think_end|>"

    if think_start in text and think_end in text:
        start_idx = text.index(think_start) + len(think_start)
        end_idx = text.index(think_end)
        thinking = text[start_idx:end_idx].strip()
        response = text[end_idx + len(think_end):].strip()
        return f"[THINKING]\n{thinking}\n[/THINKING]\n\n{response}"

    return text
