# src/common/llm.py
from __future__ import annotations
import os, json, requests
from typing import Literal, Dict, List
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

# ✅ Load .env early so env vars are available everywhere
load_dotenv()

Provider = Literal["openai", "ollama"]


class ChatError(RuntimeError): ...


class LLMClient:
    def __init__(
            self,
            provider: Provider | None = None,
            openai_model: str | None = None,
            ollama_model: str | None = None,
    ):
        self.provider: Provider = provider or os.getenv("PROVIDER", "openai")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "gemma:2b")
        print(self.provider)
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ChatError(
                    "OPENAI_API_KEY is missing. Set it in your environment or .env file."
                )
            self._client = OpenAI(api_key=api_key)
        else:
            # Ollama local server
            self._client = None
            self._ollama_url = "http://localhost:11434/api/chat"

    def chat_json(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int | None = None,
    ) -> str:
        """
        Returns a JSON string from the model (no prose).
        Uses Chat Completions + JSON mode for OpenAI.
        """
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            return content

        # Ollama (local)
        r = requests.post(
            self._ollama_url,
            json={"model": self.ollama_model, "messages": messages, "stream": False},
            timeout=60,
        )
        if r.status_code != 200:
            raise ChatError(f"Ollama HTTP {r.status_code}: {r.text}")
        return r.json().get("message", {}).get("content", "") or ""

    def run_structured(
            self,
            messages: List[Dict[str, str]],
            schema: type[BaseModel],
            *,
            temperature: float = 0.2,
            attempts: int = 3,
    ) -> BaseModel:
        """
        Calls the model, expects JSON, validates with Pydantic, and retries on parse errors.
        """
        prompt_messages = list(messages)
        last_exc: Exception | None = None
        for _ in range(attempts):
            raw = self.chat_json(prompt_messages, temperature=temperature)
            print(f""" Raw Response: {raw} """)
            try:
                data = json.loads(raw)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_exc = e
                prompt_messages.append({
                    "role": "system",
                    "content": (
                        "Your last output was invalid. Error: "
                        f"{e}. Return ONLY valid JSON exactly matching the schema."
                    ),
                })
        raise ChatError(
            f"Could not get valid structured output after {attempts} attempts. Last error: {last_exc}"
        )
