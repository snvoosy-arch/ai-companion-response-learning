"""Allowlist export of existing synthetic evaluation records; never runs a model.

The six selected cases illustrate behavior. They are spent evaluation records,
not a representative sample, training material or executable model fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    "inputs": "b821ffc5016fccbaf2991b3931db3fdf303e352a9b80e822e1f70db4ed166b2f",
    "scores": "cd688e46515ac0d98174e34ad3bfecfc62c8b39f90660a3a125f46313b897850",
    "summary": "6bde182b817662925fd47594c746bbb4175d47128b4444250124ebbfb56a6f52",
    "registry": "89ddc50c9a58b4d75a9b5b815b4c1128506001bba8d030744006e3f6831c544d",
}
CASE_SPECS = [
    {"id": "mcchat_test_0029", "title": "재고가 있어도, 먹을 수 있는지는 별개", "theme": "improved"},
    {"id": "mcchat_test_0001", "title": "접근 요청을 자연스러운 말로", "theme": "improved"},
    {"id": "mcchat_test_0012", "title": "대화가 아닌, 접근 작업의 완료", "theme": "improved"},
    {"id": "mcchat_test_0047", "title": "중단과 확인된 수량을 함께 설명", "theme": "improved"},
    {"id": "mcchat_test_0010", "title": "수량 확인을 완료 확인으로 혼동", "theme": "remaining_error"},
    {"id": "mcchat_test_0034", "title": "실행할 수 없는 대체 음식 제안", "theme": "remaining_error"},
]
MODEL_KEYS = {"baseline": "baseline_r7", "current": "minecraft_chat"}


def select_cases(inputs: list, scores: list, specs: list) -> list:
    by_id = {}
    for row in inputs:
        if row["id"] in by_id:
            raise ValueError("duplicate input")
        by_id[row["id"]] = row
    by_pair = {}
    for row in scores:
        key = (row["id"], row["adapter"])
        if key in by_pair:
            raise ValueError("duplicate output")
        by_pair[key] = row
    cases = []
    for spec in specs:
        try:
            messages = by_id[spec["id"]]["messages"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("expected one structured user message")
            source = json.loads(messages[0]["content"])
            if set(source) != {"schema_version", "message", "history", "game_context"}:
                raise ValueError("unexpected context fields")
            if source["schema_version"] != "sapphirus.minecraft.chat.v1":
                raise ValueError("unexpected context schema")
            outputs = {}
            for public, adapter in MODEL_KEYS.items():
                row = by_pair[(spec["id"], adapter)]
                if (type(row["grounded"]) is not bool or type(row["ended"]) is not bool
                        or type(row["quality"]) is not int or row["quality"] not in (0, 1, 2)
                        or row["usable"] is not (row["grounded"] and row["quality"] >= 1 and row["ended"])):
                    raise ValueError("inconsistent manual grade")
                if not isinstance(row["response"], str) or not isinstance(row["note"], str):
                    raise ValueError("invalid saved text")
                outputs[public] = {"text": row["response"], "grounded": row["grounded"],
                                   "language_quality": row["quality"], "usable": row["usable"],
                                   "normal_eos": row["ended"], "note": row["note"]}
            cases.append({"id": spec["id"], "title": spec["title"], "theme": spec["theme"],
                          "input": source, "outputs": outputs})
        except KeyError as exc:
            raise ValueError("missing selected record") from exc
    return cases


def export(paths: dict, output: Path) -> dict:
    source = {}
    for name, expected in PINS.items():
        raw = Path(paths[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {name}")
        text = raw.decode("utf-8")
        source[name] = [json.loads(line) for line in text.splitlines()] if name in {"inputs", "scores"} else json.loads(text)
    summary, registry = source["summary"], source["registry"]
    train = summary["training_result"]
    if registry["current"]["sha256"] != train["adapter_sha256"]:
        raise ValueError("current model does not match evaluated weights")
    bundle = {
        "schema_version": "sapphirus.portfolio.comparison.v1", "as_of": "2026-10-30",
        "scope": {"kind": "recorded_model_outputs", "synthetic_evaluation": True,
                  "selected_cases": len(CASE_SPECS), "representative_sample": False,
                  "spent": True, "new_model_requests": 0, "actual_game_success_evaluated": False},
        "models": {
            "baseline": {"label": "일상 Chat r7", "run": "daily_chat_sft_24030_20261018_r7",
                         "sha256": train["initial_adapter_sha256"]},
            "current": {"label": "Minecraft 대화 추가 누적본", "run": registry["current"]["run"],
                        "sha256": registry["current"]["sha256"]}},
        "training": {key: train[key] for key in ["train", "dev", "epochs", "optimizer_steps",
                    "train_loss", "dev_loss_before", "dev_loss_after", "parameters"]},
        "aggregate": {}, "provenance": dict(PINS),
        "cases": select_cases(source["inputs"], source["scores"], CASE_SPECS),
        "review": {"type": "Codex_identity_blinded_manual_review", "pairs": 62, "outputs": 124,
                   "independent_human_evaluation": False,
                   "usable_definition": "grounded and language_quality >= 1 and normal EOS"},
    }
    for group in ["minecraft", "daily"]:
        bundle["aggregate"][group] = {
            public: {key: summary[group][adapter][key] for key in
                     ["total", "grounded", "usable", "ended", "natural", "mean_quality"]}
            for public, adapter in MODEL_KEYS.items()}
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    (output / "comparison.json").write_text(text, encoding="utf-8", newline="\n")
    (output / "comparison.js").write_text("window.SAPPHIRUS_COMPARISON = " + text.rstrip() + ";\n", encoding="utf-8", newline="\n")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PINS:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = export(vars(args), args.output)
    print(json.dumps({"cases": len(bundle["cases"]), "new_model_requests": 0}))


if __name__ == "__main__":
    main()
