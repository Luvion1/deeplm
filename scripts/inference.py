import argparse
import os

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.inference.generate import load_model_for_inference, generate_text, chat


def main():
    parser = argparse.ArgumentParser(description="Inference with Deeplm model")
    parser.add_argument("--model_path", type=str, default="./deeplm_output/final", help="Path to trained model")
    parser.add_argument("--config", type=str, default="deeplm_config.yaml", help="Path to config YAML")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--thinking_mode", action="store_true", help="Enable thinking mode")
    parser.add_argument("--chat", action="store_true", help="Use chat mode")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = DeeplmConfig.from_yaml(args.config)
    else:
        config = DeeplmConfig()

    model, model_config = load_model_for_inference(args.model_path)

    if args.chat:
        messages = [
            {"role": "system", "content": "You are Deeplm, a helpful AI assistant."},
            {"role": "user", "content": args.prompt},
        ]
        response = chat(model, config, messages, max_new_tokens=args.max_tokens, temperature=args.temperature)
    else:
        response = generate_text(
            model, config, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            thinking_mode=args.thinking_mode,
        )

    print(response)


if __name__ == "__main__":
    main()
