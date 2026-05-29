from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env.black.duo.local"
DEFAULT_TRIGGER_PREFIX = "!predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black duo_smoke duo smoke를 실행합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--trigger-prefix", default=DEFAULT_TRIGGER_PREFIX)
    parser.add_argument("--console", action="store_true", help="디스코드 대신 콘솔 모드로 실행합니다.")
    parser.add_argument("--dry-run", action="store_true", help="환경만 확인하고 종료합니다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.env_file.exists():
        raise SystemExit(f"env file not found: {args.env_file}")

    env_values = dict(dotenv_values(args.env_file))
    trigger_prefix = (args.trigger_prefix or env_values.get("BOT_TRIGGER_PREFIX") or DEFAULT_TRIGGER_PREFIX).strip()
    if not trigger_prefix:
        trigger_prefix = DEFAULT_TRIGGER_PREFIX
    env_values["BOT_TRIGGER_PREFIX"] = trigger_prefix

    _apply_env(env_values.items())
    print(
        "\n".join(
            [
                f"env_file={args.env_file}",
                f"trigger_prefix={trigger_prefix!r}",
                f"duo_enabled={env_values.get('BOT_DUO_ENABLED', 'false')}",
                f"duo_partner_bot_id={_mask(env_values.get('BOT_DUO_PARTNER_BOT_ID', ''))}",
                f"duo_channel_id={env_values.get('BOT_DUO_CHANNEL_ID', '')}",
                f"duo_autostart_enabled={env_values.get('BOT_DUO_AUTOSTART_ENABLED', 'false')}",
                f"duo_autostart_channel_id={env_values.get('BOT_DUO_AUTOSTART_CHANNEL_ID', '')}",
                f"duo_autostart_prompt={'<set>' if env_values.get('BOT_DUO_AUTOSTART_PROMPT', '').strip() else '<empty>'}",
                f"console={args.console}",
                f"dry_run={args.dry_run}",
            ]
        )
    )
    if args.dry_run:
        return

    from predictive_bot.main import main as predictive_main

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    if args.console:
        sys.argv.append("--console")
    predictive_main()


def _apply_env(items: Iterable[tuple[str, str | None]]) -> None:
    for key, value in items:
        if value is None:
            continue
        os.environ[key] = value


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


if __name__ == "__main__":
    main()
