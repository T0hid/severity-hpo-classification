"""Scoring and the verification helpers (no model calls)."""
import pandas as pd


def test_baseline_scores_match_published(prepared, tmp_path):
    from hpo_severity import baseline
    baseline.score_published(tmp_path)
    s = pd.read_csv(tmp_path / "scores_full_precision.csv", encoding="utf-8-sig")
    get = lambda lvl, run, m: s[(s.level == lvl) & (s.run == run) & (s.metric == m)].iloc[0]
    assert round(get("category", "main run", "accuracy").value, 3) == 0.904
    assert round(get("category", "baseline", "accuracy").value, 3) == 0.732
    r = get("category", "main minus baseline", "accuracy")
    assert (r.mcnemar_main_right_base_wrong, r.mcnemar_main_wrong_base_right) == (179, 24)


def test_format_like():
    from hpo_severity.verify import format_like, matches
    assert format_like(90.4249, "90.4%") == "90.4%"
    assert format_like(1131.9249, "1,131.92") == "1,131.92"
    assert format_like(0.90046, "0.901") == "0.900"
    assert format_like(0.10077, "$0.101") == "$0.101"
    assert matches(1e-30, "<0.001") and not matches(0.01, "<0.001")
    assert matches(91.05, "91.1%")          # half up


def test_every_check_has_a_source():
    from hpo_severity.manuscript_numbers import CHECKS
    ids = [c["id"] for c in CHECKS]
    assert len(ids) == len(set(ids))
    assert all(c["source"] and c["expected"] for c in CHECKS)


def test_figure_style_is_reset():
    import matplotlib
    from hpo_severity import style
    matplotlib.rcParams["text.hinting"] = "default"
    matplotlib.rcParams["font.size"] = 30
    style.reset()
    assert matplotlib.rcParams["text.hinting"] == "no_hinting"
    assert matplotlib.rcParams["font.size"] == 10
    assert matplotlib.rcParams["font.sans-serif"][0] == "Arial"
    assert matplotlib.rcParams["figure.dpi"] == 200


def test_print_size():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hpo_severity import style
    style.reset()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot([0, 1], [0, 1], lw=3)
    ax.set_title("A   A long panel title", fontsize=22)
    ax.set_xlabel("x", fontsize=16)
    style.to_print_size(fig)
    assert abs(fig.get_size_inches()[0] - style.WIDTH_MM / 25.4) < 1e-9
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    assert min(sizes) >= style.FONT_MIN and max(sizes) <= style.FONT_MAX
    plt.close(fig)
