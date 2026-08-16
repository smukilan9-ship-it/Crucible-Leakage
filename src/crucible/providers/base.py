"""What Crucible needs from a model provider, and nothing more.

The audit asks one thing of a model: send this prompt, return the text. Every
provider-specific concern — authentication, retry policy, token budgets, key
rotation — lives behind that one method, which is why the rest of the package
has no idea which provider it is talking to and why the tests can cover the
whole pipeline without a network.
"""

import abc

# Ceiling on the output budget, inside the smallest context window of any model
# offered, so a doubling retry can never ask for more than a model can produce.
MAX_OUTPUT_TOKENS = 24000


class ProviderError(Exception):
    """A call could not be completed. Never raised for a bad answer, only for
    no answer: the distinction matters because a failure recorded as an answer
    is a wrong verdict that nothing downstream can detect."""


class QuotaExhausted(ProviderError):
    """Every key available to this provider is out of quota. Separate from
    ProviderError so an interface can say "come back tomorrow" rather than
    "something went wrong"."""


class Provider(abc.ABC):
    """One model provider."""

    name: str = "provider"

    @abc.abstractmethod
    async def chat(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        """Send one prompt, return the reply text. Raise rather than return
        anything empty."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        ...

    def status(self) -> dict:
        """Whatever an interface may safely show a user. Must never include a
        key, or any prefix or suffix of one."""
        return {"provider": self.name, "available": True}


def output_budget(n_columns: int) -> int:
    """Output tokens to request for a table this wide.

    Two things share this budget and only one of them is the answer. A verdict
    object runs to about sixty tokens, so the answer itself is cheap; a
    reasoning model then spends an unpredictable amount of the same budget
    thinking before it writes a single character. A budget sized for the answer
    alone comes back empty with finish_reason "length", which is how a
    36-column table failed while an 18-column one succeeded.
    """
    return min(MAX_OUTPUT_TOKENS, max(6000, 400 * n_columns))
