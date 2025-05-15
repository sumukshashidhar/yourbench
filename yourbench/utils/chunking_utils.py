from typing import Callable, Optional

import tiktoken


def split_into_token_chunks(
    text: str,
    chunk_tokens: int = 1024,
    overlap: int = 100,
    encoding_name: str = "cl100k_base",
    preprocess: Optional[Callable[[str], str]] = None,
) -> list[str]:
    """
    Splits text into token-based chunks, with optional preprocessing.

    Args:
        text (str): The input text.
        chunk_tokens (int): Max tokens per chunk.
        overlap (int): Number of overlapping tokens.
        encoding_name (str): tiktoken encoding name.
        preprocess (Optional[Callable[[str], str]]): Optional preprocessing function.

    Returns:
        list[str]: List of decoded text chunks.
    """
    if preprocess:
        text = preprocess(text)

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    stride = chunk_tokens - overlap
    return [enc.decode(tokens[i : i + chunk_tokens]) for i in range(0, len(tokens), stride)]
