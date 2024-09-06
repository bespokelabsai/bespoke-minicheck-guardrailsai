from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import os
import nltk
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
from bespokelabs import BespokeLabs


nltk.download('punkt', quiet=True)

@register_validator(name="bespokelabsai/bespokeai_factcheck", data_type="string")
class BespokeAIFactCheck(Validator):
    """Validates that the LLM-generated text is supported by the provided
    context using BespokeAI's factcheck API.

    This validator uses BespokeAI's factcheck API to evaluate the generated text
    against the provided context.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `guardrails/bespokeai_factcheck`    |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Returns supported claims            |

    Args:
        threshold (float, optional): The minimum score for a claim to be
            considered supported. Defaults to 0.5.
        split_sentences (bool, optional): Whether to split the input into
            sentences for individual evaluation. Defaults to True.
        on_fail (Optional[Callable], optional): A callable to execute when the
            validation fails. Defaults to None.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        split_sentences: bool = True,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self._split_sentences = split_sentences
        self.bl = BespokeLabs(
            auth_token=os.environ.get("BESPOKE_API_KEY"),
        )


    def _chunking_function(self, chunk: str) -> list[str]:
        return nltk.sent_tokenize(chunk)

    def _inference_local(self, model_input: Any):
        raise NotImplementedError("Local inference is not supported for BespokeAIFactCheck validator.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _inference_remote(self, model_input: Any):
        response = self.bl.minicheck.factcheck.create(
            claim=model_input["claim"],
            context=model_input["context"],
        )
        return response

    def _validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        threshold = self._threshold if "threshold" not in metadata else metadata["threshold"]
        split_sentences = self._split_sentences if "split_sentences" not in metadata else metadata["split_sentences"]
        context = metadata.get("context", "")

        if len(context) == 0:
            raise ValueError("context is required")

        if split_sentences:
            claims = nltk.sent_tokenize(value)
        else:
            claims = [value]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._inference, {"claim": claim, "context": context}) for claim in claims]
            is_supported = [future.result() for future in concurrent.futures.as_completed(futures)]

        all_claims_supported = all(
            response.support_prob >= threshold
            for response in is_supported
        )

        if all_claims_supported:
            return PassResult()
        else:
            supported_claims = [
                claim
                for claim, response in zip(claims, is_supported)
                if response.support_prob >= threshold
            ]

            return FailResult(
                error_message="Claim not supported by BespokeAI",
                fix_value=" ".join(supported_claims),
            )
