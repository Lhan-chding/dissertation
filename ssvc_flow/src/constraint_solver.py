"""Independent exhaustive one-coordinate repair solver (396 candidates at D=100)."""

from collections.abc import Sequence


def _validate_world(world, domain_size):
    if type(domain_size) is not int or domain_size < 2:
        raise ValueError("domain_size must be an integer >= 2")
    if not isinstance(world, Sequence) or len(world) != 4:
        raise ValueError("observed world must contain four integers")
    if any(type(value) is not int or not 0 <= value < domain_size for value in world):
        raise ValueError("observed world values are outside the integer domain")


def satisfies(world, cue):
    """Evaluate relationships only; these are intentionally separate from syntax V."""
    if cue is None:
        return True
    family = cue.get("family")
    if family == "duplicate_encoding":
        index, value = cue.get("known_index"), cue.get("known_value")
        if type(index) is not int or not 0 <= index < 4 or type(value) is not int:
            raise ValueError("invalid known-value cue")
        return world[index] == value
    if family == "cross_series":
        edges = cue.get("edges")
        if not isinstance(edges, (list, tuple)) or not edges:
            raise ValueError("cross_series requires nonempty edges")
        for edge in edges:
            if (
                not isinstance(edge, (list, tuple))
                or len(edge) != 3
                or any(type(value) is not int for value in edge)
                or not 0 <= edge[0] < edge[1] < 4
            ):
                raise ValueError("edges must be sorted distinct index pairs and integer totals")
        return all(world[i] + world[j] == value for i, j, value in edges)
    if family == "trend":
        a, b, c, d = world
        return a - 2 * b + c == 0 and b - 2 * c + d == 0
    raise ValueError(f"unknown constraint family: {family}")


def solve(observed, cue, domain_size=100):
    """Enumerate every admissible replacement; generator construction is not used."""
    _validate_world(observed, domain_size)
    # Validate cue even when no candidate would otherwise reach a predicate branch.
    satisfies(observed, cue)
    candidates = (
        [value if index == changed else original for index, original in enumerate(observed)]
        for changed in range(4)
        for value in range(domain_size)
        if value != observed[changed]
    )
    return [candidate for candidate in candidates if satisfies(candidate, cue)]
