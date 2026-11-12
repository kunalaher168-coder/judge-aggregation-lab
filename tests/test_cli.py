import json

from judgelab.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    r, t = tmp_path / "ratings.jsonl", tmp_path / "truth.jsonl"
    assert (
        main(
            [
                "synth",
                "--scenario",
                "mixed",
                "--items",
                "120",
                "--seed",
                "3",
                "--out",
                str(r),
                "--truth",
                str(t),
                "--meta",
                str(tmp_path / "meta.json"),
            ]
        )
        == 0
    )
    assert (
        main(["aggregate", str(r), "--method", "ds", "--out", str(tmp_path / "labels.jsonl")]) == 0
    )
    labels = [json.loads(x) for x in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 120 and {"item_id", "label", "confidence", "method"} <= set(labels[0])
    assert (
        main(
            [
                "aggregate",
                str(r),
                "--method",
                "weighted",
                "--gold",
                str(t),
                "--out",
                str(tmp_path / "w.jsonl"),
            ]
        )
        == 0
    )

    capsys.readouterr()
    assert main(["agreement", str(r), "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert -1 <= stats["overall"]["fleiss_kappa"] <= 1

    assert main(["report", str(r), "--truth", str(t), "--out", str(tmp_path / "report.md")]) == 0
    text = (tmp_path / "report.md").read_text()
    assert "Per-rater reliability" in text and "dawid_skene" in text

    assert (
        main(
            [
                "benchmark",
                "--seeds",
                "1",
                "--items",
                "100",
                "--scenarios",
                "mixed",
                "--out-md",
                str(tmp_path / "b.md"),
                "--out-json",
                str(tmp_path / "b.json"),
            ]
        )
        == 0
    )
    assert (tmp_path / "b.json").exists()


def test_cli_judge_simulated_and_routing(tmp_path):
    data = tmp_path / "support"
    assert main(["make-judge-data", "--n", "8", "--out-dir", str(data)]) == 0
    run = tmp_path / "run"
    assert (
        main(
            [
                "judge",
                "--items",
                str(data / "items.jsonl"),
                "--rubric",
                str(data / "rubric.md"),
                "--truth",
                str(data / "truth.jsonl"),
                "--backend",
                "simulated",
                "--out-dir",
                str(run),
            ]
        )
        == 0
    )
    assert (run / "routing.md").exists()
    assert (
        main(
            [
                "routing",
                str(run),
                "--truth",
                str(data / "truth.jsonl"),
                "--out",
                str(tmp_path / "r.md"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                str(run / "panel_ratings.jsonl"),
                "--truth",
                str(data / "truth.jsonl"),
                "--out",
                str(tmp_path / "panel.md"),
            ]
        )
        == 0
    )
    assert "refund_calculation" in (tmp_path / "panel.md").read_text()


def test_cli_anthropic_requires_confirmation(tmp_path):
    data = tmp_path / "support"
    main(["make-judge-data", "--n", "4", "--out-dir", str(data)])
    code = main(
        [
            "judge",
            "--items",
            str(data / "items.jsonl"),
            "--rubric",
            str(data / "rubric.md"),
            "--backend",
            "anthropic",
            "--out-dir",
            str(tmp_path / "x"),
        ]
    )
    assert code == 2 and not (tmp_path / "x").exists()
