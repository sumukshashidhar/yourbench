import tiktoken


def split_into_token_chunks(
    text: str,
    chunk_tokens: int = 1024,
    overlap: int = 100,
    encoding_name: str = "cl100k_base"
) -> list[str]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    stride = chunk_tokens - overlap
    return [enc.decode(tokens[i:i + chunk_tokens]) for i in range(0, len(tokens), stride)]
