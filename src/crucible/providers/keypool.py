"""A pool of API keys with per-key daily quotas.

Written for a specific situation: handing a demonstration to an audience on
free-tier keys that allow a small number of requests each per day. Ten keys at
twenty requests a day is two hundred requests, which is plenty for a demo and
nothing at all if one visitor can spend it.

Three rules, all of them about not being abused.

**Keys are never returned by anything.** Not by `status()`, not in an error
message, not in a log line. The pool reports how much quota is left and which
key *index* served a call; the values themselves never leave this module. A
prefix of a key is still a key to an attacker who has the rest, so not even a
prefix is exposed.

**Least-used first, not round-robin.** Round-robin spreads load evenly, which
is the wrong goal here: it burns every key to exhaustion at the same moment.
Draining the least-used key first keeps the maximum number of keys with quota
remaining, so one heavy user cannot take the whole pool down at once.

**Quota is counted before the call, not after.** A call that is started counts
against the key even if it fails. The alternative lets a failing key be retried
without limit, which is how a free tier turns into a ban.
"""

import os
import threading
import time


class KeyPool:
    def __init__(self, keys: list[str], daily_limit: int, name: str = "pool"):
        self._keys = [key.strip() for key in keys if key and key.strip()]
        self._limit = daily_limit
        self._name = name
        self._used = [0] * len(self._keys)
        self._day = self._today()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    @classmethod
    def from_environment(cls, variable: str, daily_limit: int, name: str = "pool") -> "KeyPool":
        """Read a comma or whitespace separated list of keys from the
        environment. The environment is the only supported source: a key in a
        file is a key in a commit."""
        raw = os.environ.get(variable, "")
        keys = [part for chunk in raw.split(",") for part in chunk.split() if part]
        return cls(keys, daily_limit, name)

    def acquire(self) -> tuple[int, str]:
        """Reserve one request. Returns (index, key). Raises when the whole
        pool is spent for the day."""
        with self._lock:
            self._roll_day_if_needed()
            if not self._keys:
                raise LookupError(f"{self._name}: no keys configured")
            index = min(range(len(self._keys)), key=lambda i: self._used[i])
            if self._used[index] >= self._limit:
                raise LookupError(
                    f"{self._name}: all {len(self._keys)} keys have reached their "
                    f"daily limit of {self._limit}; quota resets at midnight UTC"
                )
            self._used[index] += 1
            return index, self._keys[index]

    def remaining(self) -> int:
        with self._lock:
            self._roll_day_if_needed()
            return sum(max(0, self._limit - used) for used in self._used)

    def status(self) -> dict:
        """Safe to show a user and safe to log. Contains no key material."""
        with self._lock:
            self._roll_day_if_needed()
            return {
                "keys": len(self._keys),
                "daily_limit_per_key": self._limit,
                "remaining_today": sum(max(0, self._limit - u) for u in self._used),
                "capacity_today": self._limit * len(self._keys),
            }

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._used = [0] * len(self._keys)

    @staticmethod
    def _today() -> int:
        return int(time.time() // 86400)
