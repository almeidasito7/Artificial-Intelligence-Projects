"""
Entry point of the application.

Supports two modes:
  - CLI:  python -m src.main --user alice
  - Web:  python -m src.main --mode web [--port 8000]

The API module (FastAPI) is imported only when needed to
not penalize the CLI startup with heavy imports.
"""

from __future__ import annotations

import argparse
import sys

from src.utils import get_logger, print_answer, print_error, setup_logging

logger = get_logger("main")

def run_cli(username: str) -> None:
    """Interactive CLI loop."""
    from src.agent.router import AgentRouter
    from src.security.rls import load_user

    setup_logging()
    logger.info(f"Starting CLI as user: [bold]{username}[/bold]")

    try:
        user = load_user(username)
    except KeyError:
        print_error(f"User '{username}' not found in user_permissions.json")
        sys.exit(1)

    router = AgentRouter()

    print(f"\nBI Assistant  |  user: {username}  |  regions: {', '.join(user.regions)}")
    print("    Enter 'exit' or press Ctrl-C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = router.route(query=query, user=user)
            print_answer(
                answer=response.answer,
                source=response.source,
                cache_hit=response.cache_hit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unexpected error", exc_info=True)
            print_error(str(exc))

def run_web(host: str, port: int, debug: bool) -> None:
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    setup_logging()
    logger.info(f"Starting web server at http://{host}:{port}")

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )

def cli_entrypoint() -> None:
    """Entry point registered as console script."""
    parser = argparse.ArgumentParser(
        prog="bi-assistant",
        description="Conversational BI Assistant — Text-to-SQL + RAG + RLS + Semantic Cache",
    )
    parser.add_argument(
        "--user",
        required=False,
        default="admin",
        metavar="USERNAME",
        help="Username (must exist in user_permissions.json)",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="cli",
        help="Execution mode: 'cli' (default) or 'web'",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode / auto-reload")

    args = parser.parse_args()

    if args.mode == "web":
        run_web(host=args.host, port=args.port, debug=args.debug)
    else:
        run_cli(username=args.user)


if __name__ == "__main__":
    cli_entrypoint()
