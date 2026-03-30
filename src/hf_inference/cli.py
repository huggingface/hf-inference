import json
import os
import sys
from enum import Enum
from typing import Annotated, Optional

import typer

from openai import OpenAI


app = typer.Typer(
    help="Run inference with Hugging Face Inference Providers.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"max_content_width": 120, "help_option_names": ["-h", "--help"]},
)

ROUTER_BASE_URL = "https://router.huggingface.co/v1"


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


def _resolve_token(token: Optional[str] = None) -> str:
    if token:
        return token
    if env := os.environ.get("HF_TOKEN"):
        return env
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_path):
        return open(token_path).read().strip()
    raise typer.BadParameter("No Hugging Face token found. Set HF_TOKEN or pass --token.")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("No results found.")
        return
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def _truncate(s: str, max_len: int) -> str:
    return s[: max_len - 3] + "..." if len(s) > max_len else s


@app.command(no_args_is_help=True)
def run(
    prompt: Annotated[
        Optional[str],
        typer.Argument(
            help="Input text / prompt. Reads from stdin if omitted.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model ID on the Hugging Face Hub.",
        ),
    ] = ...,
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="Provider name or routing policy: fastest (default), cheapest, preferred.",
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            "-s",
            help="Stream output token by token.",
        ),
    ] = False,
    max_tokens: Annotated[
        Optional[int],
        typer.Option(
            "--max-tokens",
            help="Maximum number of tokens to generate.",
        ),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            "--temperature",
            help="Sampling temperature (0.0 to 2.0).",
        ),
    ] = None,
    top_p: Annotated[
        Optional[float],
        typer.Option(
            "--top-p",
            help="Nucleus sampling parameter.",
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for the conversation.",
        ),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            help="Hugging Face API token.",
        ),
    ] = None,
) -> None:
    """Run inference on a model using Hugging Face Inference Providers."""
    client = OpenAI(base_url=ROUTER_BASE_URL, api_key=_resolve_token(token))

    if prompt is not None:
        text = prompt
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        raise typer.BadParameter("No prompt provided. Pass it as an argument or pipe via stdin.")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    if ":" in model:
        model_id = model
    elif provider:
        model_id = f"{model}:{provider}"
    else:
        model_id = model
    kwargs: dict = {"model": model_id, "messages": messages, "stream": stream}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p

    if stream:
        for chunk in client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(**kwargs)
        print(response.choices[0].message.content)


@app.command(no_args_is_help=True)
def info(
    model: Annotated[str, typer.Argument(help="Model ID on the Hugging Face Hub.")] = ...,
    format: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = OutputFormat.table,
    token: Annotated[Optional[str], typer.Option("--token", help="Hugging Face API token.")] = None,
) -> None:
    """Show provider details for a model."""
    client = OpenAI(base_url=ROUTER_BASE_URL, api_key=_resolve_token(token))
    resp = client.with_raw_response.models.list()
    all_models: list[dict] = json.loads(resp.text).get("data", [])

    match = next((m for m in all_models if m.get("id") == model), None)
    if not match:
        print(f"Model '{model}' not found.")
        raise SystemExit(1)

    providers = match.get("providers") or []
    if not providers:
        print(f"No providers available for '{model}'.")
        return

    if format == OutputFormat.json:
        print(json.dumps(providers, indent=2))
        return

    headers = ["PROVIDER", "STATUS", "CONTEXT", "INPUT $/M", "OUTPUT $/M", "TOOLS", "STRUCTURED"]
    rows = []
    for p in sorted(providers, key=lambda p: p.get("provider", "")):
        pricing = p.get("pricing") or {}
        rows.append([
            p.get("provider", ""),
            p.get("status", ""),
            str(p.get("context_length", "")),
            str(pricing.get("input", "")),
            str(pricing.get("output", "")),
            "yes" if p.get("supports_tools") else "no",
            "yes" if p.get("supports_structured_output") else "no",
        ])
    _print_table(headers, rows)


@app.command("list", no_args_is_help=False)
def list_models(
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            "-p",
            help="Filter by inference provider name.",
        ),
    ] = None,
    search: Annotated[
        Optional[str],
        typer.Option(
            "--search",
            help="Filter models by name.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of models to show.",
        ),
    ] = 20,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="Output format.",
        ),
    ] = OutputFormat.table,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Only print model IDs, one per line.",
        ),
    ] = False,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            help="Hugging Face API token.",
        ),
    ] = None,
) -> None:
    """List models available via Hugging Face Inference Providers."""
    client = OpenAI(base_url=ROUTER_BASE_URL, api_key=_resolve_token(token))
    resp = client.with_raw_response.models.list()
    models: list[dict] = json.loads(resp.text).get("data", [])

    if provider:
        models = [m for m in models if any(p.get("provider") == provider for p in m.get("providers", []))]
    if search:
        q = search.lower()
        models = [m for m in models if q in m.get("id", "").lower()]
    models = models[:limit]

    if not models:
        print("No models found.")
        return

    if quiet:
        for m in models:
            print(m["id"])
        return

    def _providers_str(m: dict) -> str:
        return ", ".join(sorted(p["provider"] for p in m.get("providers", []) if p.get("provider")))

    if format == OutputFormat.json:
        print(
            json.dumps(
                [{"id": m["id"], "providers": _providers_str(m)} for m in models],
                indent=2,
            )
        )
        return

    rows = [[_truncate(m.get("id", ""), 50), _providers_str(m)] for m in models]
    _print_table(["MODEL", "PROVIDERS"], rows)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
