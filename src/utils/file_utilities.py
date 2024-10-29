# basic file utilities

def load_prompt(filename: str) -> str:
    with open(f"prompts/{filename}.md", "r") as file:
        return file.read()
