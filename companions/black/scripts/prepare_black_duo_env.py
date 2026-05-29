from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ENV = (ROOT / ".env") if (ROOT / ".env").exists() else (ROOT / ".env.example")
DEFAULT_OUTPUT_ENV = ROOT / ".env.black.duo.local"

DUO_OVERRIDES = {
    "BOT_TRIGGER_PREFIX": "!predict",
    "BOT_DUO_ENABLED": "true",
    "BOT_DUO_PARTNER_BOT_ID": "",
    "BOT_DUO_CHANNEL_ID": "",
    "BOT_DUO_MAX_TURNS_PER_BOT": "6",
    "BOT_DUO_AUTOSTART_ENABLED": "false",
    "BOT_DUO_AUTOSTART_CHANNEL_ID": "",
    "BOT_DUO_AUTOSTART_PROMPT": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black duo-test용 local env를 만듭니다.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ENV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ENV)
    parser.add_argument("--trigger-prefix", default=os.getenv("BOT_TRIGGER_PREFIX", "!predict"))
    parser.add_argument("--partner-bot-id", default=os.getenv("BOT_DUO_PARTNER_BOT_ID", ""))
    parser.add_argument("--channel-id", default=os.getenv("BOT_DUO_CHANNEL_ID", ""))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("BOT_DUO_MAX_TURNS_PER_BOT", "6")))
    parser.add_argument("--autostart-enabled", action="store_true")
    parser.add_argument("--autostart-channel-id", default=os.getenv("BOT_DUO_AUTOSTART_CHANNEL_ID", ""))
    parser.add_argument("--autostart-prompt", default=os.getenv("BOT_DUO_AUTOSTART_PROMPT", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.source.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"

    lines = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in DUO_OVERRIDES:
            seen.add(key)
            continue
        lines.append(line)

    if not args.partner_bot_id or not args.channel_id:
        raise SystemExit("partner-bot-id와 channel-id를 모두 지정해야 해.")

    duo_values = dict(DUO_OVERRIDES)
    duo_values["BOT_TRIGGER_PREFIX"] = (args.trigger_prefix or "!predict").strip() or "!predict"
    duo_values["BOT_DUO_PARTNER_BOT_ID"] = args.partner_bot_id
    duo_values["BOT_DUO_CHANNEL_ID"] = args.channel_id
    duo_values["BOT_DUO_MAX_TURNS_PER_BOT"] = str(args.max_turns)

    autostart_channel_id = args.autostart_channel_id or args.channel_id
    autostart_prompt = args.autostart_prompt.strip()
    if args.autostart_enabled:
        if not autostart_channel_id:
            raise SystemExit("autostart를 켜려면 autostart channel-id가 필요해.")
        if not autostart_prompt:
            raise SystemExit("autostart를 켜려면 autostart prompt가 필요해.")
        duo_values["BOT_DUO_AUTOSTART_ENABLED"] = "true"
        duo_values["BOT_DUO_AUTOSTART_CHANNEL_ID"] = autostart_channel_id
        duo_values["BOT_DUO_AUTOSTART_PROMPT"] = autostart_prompt
    else:
        duo_values["BOT_DUO_AUTOSTART_ENABLED"] = "false"
        duo_values["BOT_DUO_AUTOSTART_CHANNEL_ID"] = autostart_channel_id if args.autostart_channel_id else ""
        duo_values["BOT_DUO_AUTOSTART_PROMPT"] = autostart_prompt

    lines.append("")
    lines.append("# black duo test overrides")
    for key in (
        "BOT_TRIGGER_PREFIX",
        "BOT_DUO_ENABLED",
        "BOT_DUO_PARTNER_BOT_ID",
        "BOT_DUO_CHANNEL_ID",
        "BOT_DUO_MAX_TURNS_PER_BOT",
        "BOT_DUO_AUTOSTART_ENABLED",
        "BOT_DUO_AUTOSTART_CHANNEL_ID",
        "BOT_DUO_AUTOSTART_PROMPT",
    ):
        lines.append(f"{key}={duo_values[key]}")

    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        "\n".join(
            [
                f"wrote {args.output}",
                f"BOT_TRIGGER_PREFIX={duo_values['BOT_TRIGGER_PREFIX']}",
                f"BOT_DUO_ENABLED=true",
                f"BOT_DUO_PARTNER_BOT_ID={_mask(args.partner_bot_id)}",
                f"BOT_DUO_CHANNEL_ID={_mask(args.channel_id)}",
                f"BOT_DUO_MAX_TURNS_PER_BOT={args.max_turns}",
                f"BOT_DUO_AUTOSTART_ENABLED={str(args.autostart_enabled).lower()}",
                f"BOT_DUO_AUTOSTART_CHANNEL_ID={_mask(autostart_channel_id) if autostart_channel_id else ''}",
                f"BOT_DUO_AUTOSTART_PROMPT={'<set>' if autostart_prompt else '<empty>'}",
            ]
        )
    )


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


if __name__ == "__main__":
    main()
