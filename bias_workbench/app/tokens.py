"""Category-token derivation and validation for first-token classification."""

from __future__ import annotations

import re
from typing import Mapping, Sequence


def text_token_map(
    labels: Sequence[str], overrides: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """Return shortest unique first-word prefixes, honoring explicit overrides."""
    overrides = {
        str(key): str(value).strip().casefold()
        for key, value in (overrides or {}).items()
        if str(value).strip()
    }
    first_words = [re.split(r"\s+", label.strip())[0].casefold() for label in labels]
    if len(set(first_words)) != len(first_words):
        duplicates = sorted({word for word in first_words if first_words.count(word) > 1})
        raise ValueError(
            "Named-label first tokens must be distinct; repeated first word(s): "
            + ", ".join(duplicates)
        )

    tokens = []
    for label, word in zip(labels, first_words):
        override = overrides.get(label)
        if override:
            if re.search(r"\s", override):
                raise ValueError(f"Category token for '{label}' must be one token without spaces")
            if not word.startswith(override):
                raise ValueError(
                    f"Category token '{override}' is not a prefix of '{label}'s "
                    f"first word '{word}'"
                )
            tokens.append(override)
            continue

        minimum = min(2, len(word))
        prefix = word[:minimum]
        while any(other != word and other.startswith(prefix) for other in first_words):
            if len(prefix) >= len(word):
                raise ValueError(f"Could not derive a distinctive category token for '{label}'")
            prefix = word[: len(prefix) + 1]
        tokens.append(prefix)

    for index, left in enumerate(tokens):
        for other_index, right in enumerate(tokens):
            if index != other_index and (left.startswith(right) or right.startswith(left)):
                raise ValueError(
                    f"Category tokens '{left}' ({labels[index]}) and '{right}' "
                    f"({labels[other_index]}) are not distinctive"
                )
    return {label: [tokens[index]] for index, label in enumerate(labels)}


def label_token_overrides(design: Mapping) -> dict[str, str]:
    return {
        str(label["name"]): str(label.get("token") or "").strip()
        for label in design["labels"]
        if str(label.get("token") or "").strip()
    }
