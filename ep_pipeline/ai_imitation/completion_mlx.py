from pathlib import Path
from typing import Optional, Dict, Any

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler


class CompletionMLX:
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        penalty_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None

        # Sampling configuration (constructed once)
        self.sampler = make_sampler(**sampling_params) if sampling_params else None
        self.logits_processors = (
            make_logits_processors(**penalty_params) if penalty_params else None
        )

    def load(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load(self.model_id)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        verbose: bool = False,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
            verbose=verbose,
        )
        return output