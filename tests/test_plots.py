from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from compbias.plots import (
    plot_basin_map,
    plot_bifurcation,
    plot_coordination_summary,
    plot_selection_comparison,
)
from compbias.theory.coordination import BasinMap, BifurcationBranch, PhasePortrait


def _basin_map(*, singleton: bool = False) -> BasinMap:
    coordinates = np.array([0.5]) if singleton else np.array([1.0, 0.5, 0.0])
    p, q = np.meshgrid(coordinates, coordinates, indexing="xy")
    labels = np.where(
        p + q > 1.0,
        "truthful",
        np.where(p + q < 1.0, "compensatory", "separatrix"),
    )
    return BasinMap(p=p, q=q, labels=labels)


def _portrait(basins: BasinMap) -> PhasePortrait:
    return PhasePortrait(
        p=basins.p,
        q=basins.q,
        dp=basins.p * (1.0 - basins.p),
        dq=basins.q * (1.0 - basins.q),
    )


def _branch(*, crosses_critical: bool = True) -> BifurcationBranch:
    if crosses_critical:
        ratio = np.array([0.75, 0.0, 0.25, 0.5])
        positive = np.array([0.0, 1.0, 0.8, 0.0])
    else:
        ratio = np.array([0.8, 0.6])
        positive = np.zeros(2)
    return BifurcationBranch(
        beta_over_a=ratio,
        center=np.zeros_like(ratio),
        positive=positive,
        negative=-positive,
    )


def _composite_payload() -> dict[str, object]:
    return {
        "basin": _basin_map(),
        "beta_over_a": np.array([0.75, 0.0, 0.25, 0.5]),
        "predicted_positive": np.array([0.0, 1.0, 0.8, 0.0]),
        "observed_positive": np.array([0.01, 0.99, 0.79, 0.02]),
        "observed_negative": np.array([-0.01, -0.99, -0.79, -0.02]),
    }


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100
        assert image.height > 100


@pytest.mark.parametrize(
    ("name", "render"),
    [
        (
            "selection",
            lambda path: plot_selection_comparison(
                [
                    {"name": "exact", "predicted": [0.7, 0.3], "observed": [0.69, 0.31]},
                    {"name": "mirror", "predicted": [0.4, 0.6], "observed": [0.41, 0.59]},
                ],
                path,
                action_labels=("canonical", "error"),
                title="Selection contract",
                dpi=72,
            ),
        ),
        (
            "basin",
            lambda path: plot_basin_map(
                _basin_map(),
                path,
                phase_portrait=_portrait(_basin_map()),
                title="Basin contract",
                dpi=72,
            ),
        ),
        (
            "bifurcation",
            lambda path: plot_bifurcation(_branch(), path, title="Bifurcation contract", dpi=72),
        ),
        (
            "coordination",
            lambda path: plot_coordination_summary(
                {
                    7: np.array([[0.4, 0.4], [0.9, 0.9]]),
                    3: np.array([[0.6, 0.4], [0.5, 0.5]]),
                },
                path,
                title="Coordination contract",
                dpi=72,
            ),
        ),
    ],
)
def test_public_plot_apis_create_nonempty_png_without_leaking_figures(
    tmp_path: Path,
    name: str,
    render: object,
) -> None:
    sentinel = plt.figure()
    sentinel_number = sentinel.number
    try:
        before = tuple(plt.get_fignums())
        output = tmp_path / "nested" / f"{name}.png"

        returned = render(output)  # type: ignore[operator]

        assert returned == output
        _assert_png(output)
        assert tuple(plt.get_fignums()) == before
        assert sentinel_number in before
        assert plt.fignum_exists(sentinel_number)
    finally:
        plt.close(sentinel)


def test_public_plotting_does_not_mutate_caller_owned_inputs(tmp_path: Path) -> None:
    selection = [
        {"name": "b", "predicted": [0.1, 0.9], "observed": [0.2, 0.8]},
        {"name": "a", "predicted": [0.8, 0.2], "observed": [0.7, 0.3]},
    ]
    trajectories = {
        4: np.array([[0.2, 0.3], [0.8, 0.9]]),
        1: np.array([[0.7, 0.6], [0.1, 0.2]]),
    }
    composite = _composite_payload()
    selection_before = deepcopy(selection)
    trajectories_before = {key: value.copy() for key, value in trajectories.items()}
    composite_before = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in composite.items()
    }

    plot_selection_comparison(selection, tmp_path / "selection.png")
    plot_coordination_summary(trajectories, tmp_path / "rows.png")
    plot_coordination_summary(composite, tmp_path / "composite.png")

    assert selection == selection_before
    for key, value in trajectories.items():
        np.testing.assert_array_equal(value, trajectories_before[key])
    for key, value in composite.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, composite_before[key])
        else:
            assert value is composite_before[key]


def test_render_failure_cleans_up_figure_and_never_touches_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_calls: list[Figure] = []
    original_clear = Figure.clear

    def fail_save(_figure: Figure, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic renderer failure")

    def track_clear(figure: Figure, *args: object, **kwargs: object) -> None:
        clear_calls.append(figure)
        original_clear(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_save)
    monkeypatch.setattr(Figure, "clear", track_clear)
    output = tmp_path / "not-created" / "selection.png"

    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        plot_selection_comparison([0.5, 0.5], output, observed=[0.5, 0.5])

    assert clear_calls
    assert all(figure is clear_calls[0] for figure in clear_calls)
    assert not output.exists()
    assert not output.parent.exists()


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        ("", ValueError, "must not be empty"),
        ("   ", ValueError, "must not be empty"),
        (b"figure.png", TypeError, "string or path-like"),
        ("figure.jpg", ValueError, "png extension"),
        (object(), TypeError, "string or path-like"),
    ],
)
def test_output_path_rejects_unsafe_or_ambiguous_targets(
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        plot_selection_comparison([0.5, 0.5], output, observed=[0.5, 0.5])  # type: ignore[arg-type]


def test_output_path_rejects_existing_directory(tmp_path: Path) -> None:
    directory_named_png = tmp_path / "directory.png"
    directory_named_png.mkdir()

    with pytest.raises(ValueError, match="not a directory"):
        plot_selection_comparison([0.5, 0.5], directory_named_png, observed=[0.5, 0.5])


@pytest.mark.parametrize(
    ("dpi", "error", "message"),
    [
        (True, TypeError, "integer"),
        (72.0, TypeError, "integer"),
        (71, ValueError, "between 72 and 600"),
        (601, ValueError, "between 72 and 600"),
    ],
)
def test_dpi_validation(dpi: object, error: type[Exception], message: str, tmp_path: Path) -> None:
    with pytest.raises(error, match=message):
        plot_selection_comparison(
            [0.5, 0.5],
            tmp_path / "figure.png",
            observed=[0.5, 0.5],
            dpi=dpi,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("title", ["", "   ", 4])
def test_title_validation(title: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty string or None"):
        plot_bifurcation(_branch(), tmp_path / "figure.png", title=title)  # type: ignore[arg-type]


def test_selection_accepts_supported_record_and_mapping_forms(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Result:
        predicted: tuple[float, ...]
        observed: tuple[float, ...]
        name: str

    cases = [
        (
            {"wrong": 0.25, "right": 0.75},
            {"wrong": 0.2, "right": 0.8},
            None,
        ),
        (
            {"truth": [0.8, 0.2], "spurious": [0.3, 0.7]},
            {"truth": [0.7, 0.3], "spurious": [0.4, 0.6]},
            ("canonical", "error"),
        ),
    ]
    for index, (predicted, observed, labels) in enumerate(cases):
        output = tmp_path / f"paired-{index}.png"
        plot_selection_comparison(
            predicted,
            output,
            observed=observed,
            action_labels=labels,
        )
        _assert_png(output)

    records = {
        "object": Result((0.6, 0.4), (0.55, 0.45), "from-object"),
        "tuple": ([0.2, 0.8], [0.25, 0.75]),
    }
    plot_selection_comparison(records, tmp_path / "records.png")
    plot_selection_comparison(
        SimpleNamespace(predicted=[0.9, 0.1], observed=[0.85, 0.15]),
        tmp_path / "namespace.png",
    )
    plot_selection_comparison(
        [
            {"name": f"series-{index}", "predicted": [0.5, 0.5], "observed": [0.5, 0.5]}
            for index in range(4)
        ],
        tmp_path / "multi-row.png",
    )

    _assert_png(tmp_path / "records.png")
    _assert_png(tmp_path / "namespace.png")
    _assert_png(tmp_path / "multi-row.png")


@pytest.mark.parametrize(
    ("results", "observed", "message"),
    [
        ({}, {}, "must not be empty"),
        ({"a": [0.5, 0.5]}, {"b": [0.5, 0.5]}, "identical keys"),
        ({1: [0.5, 0.5]}, {1: [0.5, 0.5]}, "non-empty strings"),
        ({"predicted": [0.5, 0.5]}, None, "contain 'observed'"),
        ([], None, "must not be empty"),
        (3, None, "record, mapping, or iterable"),
        ([object()], None, "expose predicted and observed"),
        ([([0.5, 0.5], [0.5, 0.5], "extra")], None, "expose predicted and observed"),
    ],
)
def test_selection_rejects_invalid_input_structures(
    results: object, observed: object | None, message: str, tmp_path: Path
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plot_selection_comparison(
            results,
            tmp_path / "figure.png",
            observed=observed,
        )


@pytest.mark.parametrize(
    ("probability", "message"),
    [
        ("not-numeric", "finite probability vector"),
        ([[0.5, 0.5]], "one-dimensional"),
        ([], "one-dimensional"),
        ([np.nan, np.nan], "only finite"),
        ([-0.1, 1.1], r"lie in \[0, 1\]"),
        ([0.2, 0.2], "sum to one"),
    ],
)
def test_selection_rejects_invalid_probability_vectors(
    probability: object, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_selection_comparison(
            probability,
            tmp_path / "figure.png",
            observed=[0.5, 0.5],
        )


def test_selection_rejects_shape_name_and_action_label_ambiguities(tmp_path: Path) -> None:
    invalid_calls = [
        (
            [
                {"name": "x", "predicted": [1.0], "observed": [1.0]},
                {"name": "y", "predicted": [0.5, 0.5], "observed": [0.5, 0.5]},
            ],
            {},
            "same number of actions",
        ),
        (
            [{"name": "same", "predicted": [0.5, 0.5], "observed": [0.5, 0.5]}] * 2,
            {},
            "names must be unique",
        ),
        (
            {"name": " ", "predicted": [0.5, 0.5], "observed": [0.5, 0.5]},
            {},
            "names must be non-empty",
        ),
        ([0.5, 0.5], {"observed": [0.5, 0.5], "action_labels": "ab"}, "exactly 2"),
        ([0.5, 0.5], {"observed": [0.5, 0.5], "action_labels": ["a"]}, "exactly 2"),
        (
            [0.5, 0.5],
            {"observed": [0.5, 0.5], "action_labels": ["a", " "]},
            "non-empty strings",
        ),
        (
            [0.5, 0.5],
            {"observed": [0.5, 0.5], "action_labels": ["a", "a"]},
            "must be unique",
        ),
    ]
    for results, kwargs, message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            plot_selection_comparison(results, tmp_path / "figure.png", **kwargs)


def test_basin_plot_accepts_sorted_or_singleton_mesh_and_optional_portrait(
    tmp_path: Path,
) -> None:
    basins = _basin_map()
    singleton = _basin_map(singleton=True)

    plot_basin_map(basins, tmp_path / "with-field.png", phase_portrait=_portrait(basins))
    plot_basin_map(singleton, tmp_path / "singleton.png")

    _assert_png(tmp_path / "with-field.png")
    _assert_png(tmp_path / "singleton.png")


@pytest.mark.parametrize(
    ("basins", "message"),
    [
        (object(), "must be a BasinMap"),
        (BasinMap(p=[0.5], q=[0.5], labels=["truthful"]), "shape-matched matrices"),
        (
            BasinMap(p=[[0.0, 1.0]], q=[[0.0], [1.0]], labels=[["truthful"]]),
            "shape-matched matrices",
        ),
        (
            BasinMap(p=np.empty((0, 0)), q=np.empty((0, 0)), labels=np.empty((0, 0))),
            "non-empty",
        ),
        (
            BasinMap(p=[[0.0, np.nan]], q=[[0.0, 0.0]], labels=[["truthful"]] * 2),
            "only finite",
        ),
        (
            BasinMap(p=[[0.0, 1.1]], q=[[0.0, 0.0]], labels=[["truthful"]] * 2),
            "closed unit square",
        ),
        (
            BasinMap(
                p=[[0.0, 1.0], [0.1, 1.0]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                labels=[["truthful"] * 2] * 2,
            ),
            "constant down each mesh column",
        ),
        (
            BasinMap(
                p=[[0.0, 1.0], [0.0, 1.0]],
                q=[[0.0, 0.1], [1.0, 1.0]],
                labels=[["truthful"] * 2] * 2,
            ),
            "constant across each mesh row",
        ),
        (
            BasinMap(
                p=[[0.5, 0.5], [0.5, 0.5]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                labels=[["truthful"] * 2] * 2,
            ),
            "must be unique",
        ),
        (
            BasinMap(
                p=[[0.0, 1.0], [0.0, 1.0]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                labels=["truthful", "compensatory"],
            ),
            "labels must match",
        ),
        (
            BasinMap(
                p=[[0.0, 1.0], [0.0, 1.0]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                labels=[["truthful", "mystery"], ["truthful", "compensatory"]],
            ),
            "unknown basin labels: mystery",
        ),
    ],
)
def test_basin_plot_rejects_invalid_meshes(basins: object, message: str, tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plot_basin_map(basins, tmp_path / "figure.png")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("portrait", "message"),
    [
        (object(), "PhasePortrait or None"),
        (
            PhasePortrait(p=[[0.0]], q=[[0.0]], dp=[[0.0]], dq=[[0.0]]),
            "must match the basin mesh shape",
        ),
        (
            PhasePortrait(
                p=[[1.0, 0.0], [1.0, 0.0]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                dp=[[np.inf, 0.0], [0.0, 0.0]],
                dq=np.zeros((2, 2)),
            ),
            "only finite",
        ),
        (
            PhasePortrait(
                p=[[0.9, 0.0], [0.9, 0.0]],
                q=[[0.0, 0.0], [1.0, 1.0]],
                dp=np.zeros((2, 2)),
                dq=np.zeros((2, 2)),
            ),
            "same coordinates",
        ),
    ],
)
def test_basin_plot_rejects_misaligned_phase_portraits(
    portrait: object, message: str, tmp_path: Path
) -> None:
    basins = BasinMap(
        p=[[1.0, 0.0], [1.0, 0.0]],
        q=[[0.0, 0.0], [1.0, 1.0]],
        labels=[["separatrix", "compensatory"], ["truthful", "separatrix"]],
    )
    with pytest.raises((TypeError, ValueError), match=message):
        plot_basin_map(
            basins,
            tmp_path / "figure.png",
            phase_portrait=portrait,  # type: ignore[arg-type]
        )


def test_bifurcation_plot_handles_branches_on_both_sides_of_critical_ratio(
    tmp_path: Path,
) -> None:
    plot_bifurcation(_branch(), tmp_path / "crossing.png")
    plot_bifurcation(_branch(crosses_critical=False), tmp_path / "above.png")

    _assert_png(tmp_path / "crossing.png")
    _assert_png(tmp_path / "above.png")


@pytest.mark.parametrize(
    ("branch", "message"),
    [
        (object(), "must be a BifurcationBranch"),
        (
            BifurcationBranch(
                beta_over_a=[[0.0, 0.5]],
                center=[[0.0, 0.0]],
                positive=[[1.0, 0.0]],
                negative=[[-1.0, 0.0]],
            ),
            "one-dimensional",
        ),
        (
            BifurcationBranch(beta_over_a=[0.0], center=[0.0], positive=[1.0], negative=[-1.0]),
            "length of at least two",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, 0.5],
                center=[0.0],
                positive=[1.0, 0.0],
                negative=[-1.0, 0.0],
            ),
            "share a length",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, np.nan],
                center=[0.0, 0.0],
                positive=[1.0, 0.0],
                negative=[-1.0, 0.0],
            ),
            "only finite",
        ),
        (
            BifurcationBranch(
                beta_over_a=[-0.1, 0.5],
                center=[0.0, 0.0],
                positive=[1.0, 0.0],
                negative=[-1.0, 0.0],
            ),
            "must be nonnegative",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.5, 0.5],
                center=[0.0, 0.0],
                positive=[0.0, 0.0],
                negative=[0.0, 0.0],
            ),
            "must be unique",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, 0.5],
                center=[0.0, 0.0],
                positive=[1.1, 0.0],
                negative=[-1.1, 0.0],
            ),
            r"lie in \[0, 1\]",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, 0.5],
                center=[0.0, 0.0],
                positive=[1.0, 0.0],
                negative=[-0.9, 0.1],
            ),
            r"lie in \[-1, 0\]",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, 0.5],
                center=[0.1, 0.0],
                positive=[1.0, 0.0],
                negative=[-1.0, 0.0],
            ),
            "center branch must be zero",
        ),
        (
            BifurcationBranch(
                beta_over_a=[0.0, 0.5],
                center=[0.0, 0.0],
                positive=[1.0, 0.0],
                negative=[-0.9, 0.0],
            ),
            "branches must be symmetric",
        ),
    ],
)
def test_bifurcation_plot_rejects_invalid_branches(
    branch: object, message: str, tmp_path: Path
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plot_bifurcation(branch, tmp_path / "figure.png")  # type: ignore[arg-type]


def test_coordination_summary_accepts_rows_histories_and_columnar_records(
    tmp_path: Path,
) -> None:
    history = [
        SimpleNamespace(truthful_perception_probability=0.2, canonical_reasoning_probability=0.3),
        SimpleNamespace(truthful_perception_probability=0.8, canonical_reasoning_probability=0.9),
    ]
    rows = [
        {"seed": 2, "history": history},
        {"seed": 1, "basin_label": "compensatory", "equilibrium_mode": "stable"},
    ]
    columnar = {
        "seed": [4, 3],
        "trajectory": [
            {"p": [0.2, 0.8], "q": [0.3, 0.9]},
            np.array([[0.8, 0.7], [0.2, 0.1]]),
        ],
        "endpoint_label": ["truthful", "compensatory"],
    }

    plot_coordination_summary(rows, tmp_path / "rows.png")
    plot_coordination_summary(columnar, tmp_path / "columnar.png")
    plot_coordination_summary(
        SimpleNamespace(seed=9, basin="truthful", equilibrium="stable"),
        tmp_path / "object.png",
    )

    _assert_png(tmp_path / "rows.png")
    _assert_png(tmp_path / "columnar.png")
    _assert_png(tmp_path / "object.png")


def test_coordination_composite_accepts_sorted_and_noncritical_ratios(tmp_path: Path) -> None:
    payload = _composite_payload()
    plot_coordination_summary(payload, tmp_path / "crossing.png")
    payload["beta_over_a"] = np.array([0.8, 0.6])
    payload["predicted_positive"] = np.array([0.0, 0.0])
    payload["observed_positive"] = np.array([0.01, 0.02])
    payload["observed_negative"] = np.array([-0.01, -0.02])
    plot_coordination_summary(payload, tmp_path / "above.png")

    _assert_png(tmp_path / "crossing.png")
    _assert_png(tmp_path / "above.png")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ({}, "must not be empty"),
        ([], "must not be empty"),
        ("invalid", "record, mapping, or iterable"),
        (7, "record, mapping, or iterable"),
        ({"seed": []}, "must not be empty"),
        ({"seed": [1, 2], "basin": ["truthful"]}, "matching lengths"),
        ([{"seed": 1, "basin": "truthful"}, {"seed": 1, "basin": "truthful"}], "unique"),
        ([{"seed": True, "basin": "truthful"}], "non-negative integers"),
        ([{"seed": -1, "basin": "truthful"}], "non-negative integers"),
        ([{"seed": 1, "basin": " "}], "non-empty strings"),
        ([{"seed": 1}], "needs a basin/equilibrium label or trajectory"),
        ([{"seed": index, "basin": "truthful"} for index in range(101)], "at most 100"),
    ],
)
def test_coordination_summary_rejects_invalid_record_collections(
    records: object, message: str, tmp_path: Path
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plot_coordination_summary(records, tmp_path / "figure.png")


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        ("invalid", "non-empty iterable"),
        ([], r"shape \(time, 2\)"),
        ([{"p": 0.5}], "must expose perception and reasoning"),
        ([{"p": "bad", "q": 0.5}], "two finite probabilities"),
        ([{"p": np.inf, "q": 0.5}], "two finite probabilities"),
        ([{"p": 1.1, "q": 0.5}], r"lie in \[0, 1\]"),
        ({"p": [0.1], "q": [0.2, 0.3]}, "aligned vectors"),
        ({"p": ["bad"], "q": [0.2]}, "must be numeric"),
        ([[0.1, 0.2, 0.3]], r"shape \(time, 2\)"),
        ([[0.1, np.nan]], "only finite"),
        ([[0.1, 1.2]], r"lie in \[0, 1\]"),
    ],
)
def test_coordination_summary_rejects_invalid_trajectories(
    trajectory: object, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_coordination_summary(
            [{"seed": 1, "trajectory": trajectory}],
            tmp_path / "figure.png",
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"basin": object()}, "basin must be a BasinMap"),
        ({"beta_over_a": ["bad", "data"]}, "finite one-dimensional array"),
        ({"beta_over_a": [[0.0, 0.5]]}, "one-dimensional array"),
        ({"beta_over_a": [0.0]}, "length at least two"),
        ({"beta_over_a": [0.0, np.inf]}, "only finite"),
        ({"predicted_positive": [0.0, 0.5, 1.0]}, "identical shapes"),
        ({"beta_over_a": [-0.1, 0.0, 0.25, 0.5]}, "must be nonnegative"),
        ({"predicted_positive": [1.1, 1.0, 0.8, 0.0]}, r"lie in \[0, 1\]"),
        ({"observed_positive": [1.1, 0.99, 0.79, 0.02]}, r"lie in \[0, 1\]"),
        ({"observed_negative": [0.1, -0.99, -0.79, -0.02]}, r"lie in \[-1, 0\]"),
        ({"beta_over_a": [0.5, 0.0, 0.25, 0.5]}, "must be unique"),
    ],
)
def test_coordination_composite_rejects_invalid_payloads(
    updates: dict[str, object], message: str, tmp_path: Path
) -> None:
    payload = _composite_payload()
    payload.update(updates)
    with pytest.raises((TypeError, ValueError), match=message):
        plot_coordination_summary(payload, tmp_path / "figure.png")
