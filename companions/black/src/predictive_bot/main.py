from __future__ import annotations

import argparse
import asyncio
import traceback

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some runtime lanes
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

from predictive_bot.config import AppConfig
from predictive_bot.discord_app.bot import run_discord_bot
from predictive_bot.factory import build_engine, build_speech_runtime_for_bot
from predictive_bot.startup_lock import StartupLockError, acquire_startup_lock


def main() -> None:
    load_dotenv()
    print("[startup] dotenv loaded", flush=True)

    parser = argparse.ArgumentParser(
        description="예측/목표 중심 코어와 선택적 LLM 문장화 레이어를 가진 디스코드 봇"
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="디스코드 대신 로컬 콘솔 채팅 모드로 실행합니다.",
    )
    args = parser.parse_args()
    print(f"[startup] args console={args.console}", flush=True)

    config = AppConfig.from_env()
    print(
        "[startup] config loaded "
        f"persona={config.bot_persona!r} "
        f"model_alias={config.black_model_alias!r} "
        f"strict_llm_only={config.strict_llm_only} "
        f"duo_enabled={config.duo_mode_enabled} "
        f"duo_channel={config.duo_channel_id!r} "
        f"autostart_enabled={config.duo_autostart_enabled} "
        f"runtime_enabled={config.runtime_state_enabled} "
        f"startup_lock_enabled={config.startup_lock_enabled}",
        flush=True,
    )

    try:
        with acquire_startup_lock(
            config.startup_lock_path,
            enabled=config.startup_lock_enabled,
            bot_name=config.runtime_bot_name,
        ):
            print(f"[startup] startup lock acquired path={config.startup_lock_path}", flush=True)
            engine = build_engine(config)
            speech_runtime = build_speech_runtime_for_bot(config)
            print("[startup] engine built", flush=True)
            try:
                if args.console:
                    print("[startup] entering console mode", flush=True)
                    asyncio.run(_run_console(engine))
                    return

                if not config.discord_bot_token:
                    raise SystemExit("DISCORD_BOT_TOKEN이 없습니다. 로컬 테스트는 --console로 실행하세요.")

                print("[startup] starting discord bot", flush=True)
                asyncio.run(run_discord_bot(config, engine, speech_runtime=speech_runtime))
            finally:
                state_store = getattr(engine, "state_store", None)
                if state_store is not None:
                    try:
                        state_store.close()
                        print("[startup] state_store closed", flush=True)
                    except Exception:
                        print("[startup] state_store close failed", flush=True)
                        traceback.print_exc()
    except StartupLockError as exc:
        print(f"[startup] startup lock error: {exc}", flush=True)
        raise SystemExit(str(exc)) from exc
    except Exception:
        print("[startup] unhandled exception before shutdown", flush=True)
        traceback.print_exc()
        raise


async def _run_console(engine) -> None:
    print("콘솔 모드입니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    user_id = "console-user"

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        result = await engine.respond(user_id=user_id, text=user_text)
        print(f"bot> {result.reply}")
        print(f"audit> {result.audit_record.format_for_log()}")


if __name__ == "__main__":
    main()
