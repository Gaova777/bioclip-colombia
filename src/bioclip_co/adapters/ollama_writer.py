from langchain_ollama import ChatOllama


class OllamaWriter:
    """FieldGuideWriter on a local model served by Ollama."""

    def __init__(self, model: str = "qwen3:4b-instruct", base_url: str = "http://localhost:11434", num_ctx: int = 2048):
        # 2048 tokens fit a description and its card
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0, reasoning=False, num_ctx=num_ctx)

    def card(self, name: str, common: str, notes: str) -> str:
        prompt = (
            f"Write a field guide entry for {name}{f' ({common})' if common else ''} in at most 70 words, "
            "plain English, no headings, no lists. Mention how to recognize it and where it lives. "
            f"Use only these notes and don't add facts that aren't in them:\n\n{notes}"
        )
        return self.llm.invoke(prompt).content.strip()
