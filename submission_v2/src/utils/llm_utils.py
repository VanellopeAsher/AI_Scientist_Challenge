import os
import re
import openai
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import dotenv_values

from .vllm import infer_vllm

# Load environment variables: first from system environment (for Docker), then from .env file (for local dev)
# Find .env file in the retrieval directory (parent of src directory)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
retrieval_dir = os.path.dirname(os.path.dirname(current_file_dir))  # Go up from src/utils to retrieval
env_file_path = os.path.join(retrieval_dir, ".env")

# Try to load from .env file if it exists
env_vars = {}
if os.path.exists(env_file_path):
    env_vars = dotenv_values(env_file_path)

# Set openai credentials (prefer system env, then .env file)
openai.api_key = os.getenv("OPENAI_API_KEY") or env_vars.get("OPENAI_API_KEY")

# Model credentials must be provided via SCI_* environment variables
# Prefer system environment variables (for Docker), fallback to .env file (for local dev)
SCI_MODEL_API_KEY = os.getenv("SCI_MODEL_API_KEY") or env_vars.get("SCI_MODEL_API_KEY")
SCI_MODEL_BASE_URL = os.getenv("SCI_MODEL_BASE_URL") or env_vars.get("SCI_MODEL_BASE_URL")

# Validate that required environment variables are present
if not SCI_MODEL_API_KEY:
    raise ValueError(f"SCI_MODEL_API_KEY is required but not found. Please set it as an environment variable or add it to {env_file_path}")
if not SCI_MODEL_BASE_URL:
    raise ValueError(f"SCI_MODEL_BASE_URL is required but not found. Please set it as an environment variable or add it to {env_file_path}")


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
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """
    This function actually calls the OpenAI-compatible API
    Models such as 'gpt-4o-mini' and 'deepseek-chat' are available
    :param json_data:
    :param use_deepseek: Whether to use DeepSeek API
    :param max_retries: Maximum number of retry attempts for API errors
    :param retry_delay: Delay in seconds between retries
    :return: dict with 'response', 'usage', 'cost' keys
    """
    import time
    
    if use_deepseek:
        # Use DeepSeek API
        # Note: These variables are validated at module load time from .env file
        if not SCI_MODEL_API_KEY:
            raise ValueError("SCI_MODEL_API_KEY is not set in .env file. Please add it to your .env file.")
        if not SCI_MODEL_BASE_URL:
            raise ValueError("SCI_MODEL_BASE_URL is not set in .env file. Please add it to your .env file.")
        # Use deepseek-chat as the model name
        model_name = "deepseek-chat"
        openai_client = openai.OpenAI(
            api_key=SCI_MODEL_API_KEY,
            base_url=SCI_MODEL_BASE_URL
        )
        
        # Retry logic for API calls
        last_error = None
        for attempt in range(max_retries):
            try:
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
            except openai.APIError as e:
                last_error = e
                status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                if status_code == 500 or status_code == 502 or status_code == 503:
                    # Server errors - retry
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                        print(f"  ⚠️  API错误 (状态码 {status_code}): {str(e)[:100]}")
                        print(f"     等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                # For other errors or max retries reached, raise
                print(f"  ❌ API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:200]}")
                raise
            except Exception as e:
                last_error = e
                print(f"  ❌ 意外错误 (尝试 {attempt + 1}/{max_retries}): {str(e)[:200]}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"     等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                raise
        
        # If all retries failed
        raise Exception(f"API调用失败，已重试 {max_retries} 次。最后错误: {last_error}")
    else:
        # Use OpenAI API
        openai_client = openai.OpenAI()
        
        # Retry logic for API calls
        last_error = None
        for attempt in range(max_retries):
            try:
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
            except openai.APIError as e:
                last_error = e
                status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                if status_code in [500, 502, 503, 429]:
                    # Server errors or rate limit - retry
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                        print(f"  ⚠️  API错误 (状态码 {status_code}): {str(e)[:100]}")
                        print(f"     等待 {wait_time:.1f} 秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                # For other errors or max retries reached, raise
                print(f"  ❌ API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:200]}")
                raise
            except Exception as e:
                last_error = e
                print(f"  ❌ 意外错误 (尝试 {attempt + 1}/{max_retries}): {str(e)[:200]}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"     等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                raise
        
        # If all retries failed
        raise Exception(f"API调用失败，已重试 {max_retries} 次。最后错误: {last_error}")
