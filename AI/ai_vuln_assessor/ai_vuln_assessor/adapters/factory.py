from __future__ import annotations
from .openai_adapter import OpenAIChatAdapter
from .anthropic_adapter import AnthropicMessagesAdapter
from .openai_compatible_adapter import OpenAICompatibleAdapter
from .simple_http_adapter import SimpleHTTPJSONAdapter, SimpleHTTPFormAdapter

def build_adapter(args):
    if args.adapter == "openai_chat":
        return OpenAIChatAdapter(args.model_name, args.api_key, args.base_url, args.max_tokens)
    if args.adapter == "anthropic_messages":
        return AnthropicMessagesAdapter(args.model_name, args.api_key, args.max_tokens)
    if args.adapter == "openai_compatible":
        return OpenAICompatibleAdapter(args.base_url, args.model_name, args.api_key, args.max_tokens)
    if args.adapter == "simple_http_json":
        return SimpleHTTPJSONAdapter(args.base_url, args.json_prompt_key, args.response_json_path, args.http_method)
    if args.adapter == "simple_http_form":
        return SimpleHTTPFormAdapter(args.base_url, args.form_prompt_key, args.response_json_path, args.http_method)
    raise ValueError(f"Unsupported adapter: {args.adapter}")
