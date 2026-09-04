from __future__ import annotations

from .contracts import SemanticFeature


def features(**values: str) -> tuple[SemanticFeature, ...]:
    return tuple(SemanticFeature(axis=axis, value=value) for axis, value in values.items())


def feature_map(values: tuple[SemanticFeature, ...]) -> dict[str, str]:
    if not isinstance(values, tuple):
        raise ValueError("semantic features must be a tuple")
    mapped: dict[str, str] = {}
    for feature in values:
        if not isinstance(feature, SemanticFeature):
            raise ValueError("semantic feature has an invalid type")
        if (
            not feature.axis
            or not feature.axis.replace("_", "").isalnum()
            or feature.axis.casefold() != feature.axis
        ):
            raise ValueError(f"invalid semantic axis: {feature.axis!r}")
        if not feature.value:
            raise ValueError(f"empty semantic value: {feature.axis}")
        if feature.axis in mapped:
            raise ValueError(f"duplicate semantic axis: {feature.axis}")
        mapped[feature.axis] = feature.value
    return mapped


def replace_features(
    values: tuple[SemanticFeature, ...],
    **updates: str,
) -> tuple[SemanticFeature, ...]:
    mapped = feature_map(values)
    mapped.update(updates)
    return features(**mapped)
