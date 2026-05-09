MAX_TOKENS = 350  # conservative

def normalize_prompt(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    max_chars = max_tokens * 4  
    return prompt[:max_chars]
