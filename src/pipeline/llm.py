"""
LLM client using Groq with automatic model fallback.

If one model hits the rate limit, automatically switches to the next
available model. Falls through the entire list before giving up.

Models ordered best → fastest (fallback chain):
- openai/gpt-oss-120b        (best quality, 120B)
- llama-3.3-70b-versatile    (strong, 70B)
- moonshotai/kimi-k2-instruct (strong reasoning)
- moonshotai/kimi-k2-instruct-0905
- qwen/qwen3-32b             (solid reasoning, 32B)
- groq/compound              (Groq compound)
- openai/gpt-oss-20b         (mid-size)
- openai/gpt-oss-safeguard-20b
- groq/compound-mini
- meta-llama/llama-4-scout-17b-16e-instruct
- canopylabs/orpheus-v1-english
- canopylabs/orpheus-arabic-saudi
- meta-llama/llama-prompt-guard-2-86m
- meta-llama/llama-prompt-guard-2-22m
- llama-3.1-8b-instant       (last resort, fastest)
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# Models ordered best quality → fastest. Falls back down the list on rate limit.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",                          # best quality — 120B
    "llama-3.3-70b-versatile",                      # strong — 70B
    "moonshotai/kimi-k2-instruct",                  # strong reasoning
    "moonshotai/kimi-k2-instruct-0905",             # kimi variant
    "qwen/qwen3-32b",                               # solid reasoning — 32B
    "groq/compound",                                # Groq compound
    "openai/gpt-oss-20b",                          # mid-size — 20B
    "openai/gpt-oss-safeguard-20b",                # safeguard variant — 20B
    "groq/compound-mini",                           # compact compound
    "meta-llama/llama-4-scout-17b-16e-instruct",   # fast scout
    "canopylabs/orpheus-v1-english",               # domain-specific (English)
    "canopylabs/orpheus-arabic-saudi",             # domain-specific (Arabic)
    "meta-llama/llama-prompt-guard-2-86m",         # tiny — 86M
    "meta-llama/llama-prompt-guard-2-22m",         # tiny — 22M
    "llama-3.1-8b-instant",                        # last resort — fastest, 8B
]

# Audio transcription models — used separately via transcribe(), not in fallback chain.
WHISPER_MODELS = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]


class GeminiClient:
    """
    Wrapper around the Groq API with automatic model fallback.

    Always starts from the best available model (openai/gpt-oss-120b) and
    falls down to faster/smaller models only when rate-limited or blocked.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get a free key from https://console.groq.com "
                "and add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.models = list(FALLBACK_MODELS)
        self.current_model_idx = 0

    @property
    def current_model(self):
        return self.models[self.current_model_idx]

    def _try_generate(self, prompt, model, temperature, max_tokens):
        """Try a single model. Returns response or raises."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate(
        self,
        prompt: str,
        use_quality_model: bool = False,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> str:
        """
        Send a prompt to Groq. Always tries the best model first, then
        falls through to smaller/faster models on rate limit or block.

        Args:
            prompt: the text prompt
            use_quality_model: ignored (kept for compatibility), always starts
                from the best available model
            temperature: controls randomness
            max_retries: retries per model before moving to fallback

        Returns:
            the model's text response
        """
        errors = []

        for model_idx, model in enumerate(self.models):
            for attempt in range(max_retries):
                try:
                    result = self._try_generate(prompt, model, temperature, 4096)
                    if model_idx != 0:
                        print(f"  [OK] Using {model} (fell back from best model due to rate limit)")
                    self.current_model_idx = model_idx
                    return result

                except Exception as e:
                    error_str = str(e)
                    is_skip = (
                        "429" in error_str
                        or "rate_limit" in error_str.lower()
                        or "403" in error_str
                        or "permission" in error_str.lower()
                    )

                    if is_skip:
                        print(f"  [!] {model} rate-limited, trying next best model...")
                        errors.append(f"{model}: rate limited")
                        break  # skip retries, move to next model
                    else:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"  Groq error ({model}): {e}. Retry in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            errors.append(f"{model}: {e}")
                            break  # move to next model

        raise Exception(f"All models exhausted. Errors: {'; '.join(errors)}")

    def classify(self, prompt: str) -> str:
        """Quick classification — starts from best model, falls back if needed."""
        return self.generate(prompt, temperature=0.0)

    def analyze(self, prompt: str) -> str:
        """Detailed analysis — starts from best model, falls back if needed."""
        return self.generate(prompt, temperature=0.1)

    def transcribe(self, audio_file_path: str, model: str = "whisper-large-v3") -> str:
        """
        Transcribe audio using a Whisper model.

        Args:
            audio_file_path: path to the audio file
            model: 'whisper-large-v3' (default, higher accuracy)
                   or 'whisper-large-v3-turbo' (faster)

        Returns:
            transcribed text
        """
        if model not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid Whisper model '{model}'. Choose from: {WHISPER_MODELS}"
            )
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
        return transcription.text


if __name__ == "__main__":
    client = GeminiClient()
    response = client.classify("Is this a comparison question? 'What are Apple's risk factors?'")
    print(f"Response: {response}")