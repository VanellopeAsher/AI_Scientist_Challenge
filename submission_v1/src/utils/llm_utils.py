import os
import re
import openai
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


from .vllm import infer_vllm

# Set openai credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set deepseek credentials
# Read API key from environment variable SCI_MODEL_API_KEY (required by competition rules)
DEEPSEEK_API_KEY = os.environ.get("SCI_MODEL_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def extract_ids(papers_string):
    pattern = r"([a-f0-9]{40}):"
    matches = re.findall(pattern, papers_string)
    return matches


def parse_papers_reranking(combined_papers):
    # Check if the input is NaN
    if pd.isna(combined_papers):
        return np.nan
    pattern = r"ID: (.*?) - Title: (.*?)"
    matches = re.findall(pattern, combined_papers)
    return [f"{match[0].strip()} {match[1]}" for match in matches]


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        # print(matches)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def run_llm_api(
    json_data,
    gen_engine="gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> dict:
    """
    This function actually calls the OpenAI API
    Models such as 'gpt-4o-mini', 'deepseek-chat', and 'llama-*' are available
    :param json_data:
    :return: dict with 'response', 'usage', 'cost' keys (for backward compatibility, can return str if needed)
    """
    if "deepseek" in gen_engine.lower():
        return run_openai_api(json_data, gen_engine, max_tokens, temperature, use_deepseek=True)
    elif "gpt" in gen_engine:
        return run_openai_api(json_data, gen_engine, max_tokens, temperature)
    elif "llama" in gen_engine:
        response = infer_vllm(
            prompt=json_data["prompt"],
            max_tokens=max_tokens,
            end_point=gen_engine,
            temperature=temperature,
        )
        # For vllm, we don't have usage info, return dict with response only
        return {
            "response": response,
            "usage": None,
            "cost": 0
        }


def run_openai_api(
    json_data,
    gen_engine="gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0.2,
    use_deepseek: bool = False,
) -> dict:
    """
    This function actually calls the OpenAI-compatible API
    Models such as 'gpt-4o-mini' and 'deepseek-chat' are available
    :param json_data:
    :param use_deepseek: Whether to use DeepSeek API
    :return: dict with 'response', 'usage', 'cost' keys
    """
    if use_deepseek:
        # Use DeepSeek API
        if not DEEPSEEK_API_KEY:
            raise ValueError("SCI_MODEL_API_KEY is not set. Please set it as an environment variable")
        # Use deepseek-chat as the model name
        model_name = "deepseek-chat"
        openai_client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        completion = openai_client.chat.completions.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": f"{json_data['system_prompt']}"},
                {"role": "user", "content": f"{json_data['prompt']}"},
            ],
        )
        
        # DeepSeek pricing: $0.14 per 1M input tokens, $0.28 per 1M output tokens (as of 2024)
        usage = completion.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        # Calculate cost (in USD)
        cost = (input_tokens / 1_000_000 * 0.14) + (output_tokens / 1_000_000 * 0.28)
        
        return {
            "response": completion.choices[0].message.content,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": usage.total_tokens
            },
            "cost": cost
        }
    else:
        # Use OpenAI API
        openai_client = openai.OpenAI()
        completion = openai_client.chat.completions.create(
            model=gen_engine,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": f"{json_data['system_prompt']}"},
                {"role": "user", "content": f"{json_data['prompt']}"},
            ],
        )
        
        usage = completion.usage
        return {
            "response": completion.choices[0].message.content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "cost": 0  # OpenAI cost calculation would need model-specific pricing
        }
