"""Command-line entry points for the standalone audio module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from research_kb.audio_web import serve_audio_review
from research_kb.config import AppSettings

app = typer.Typer(name="voice-to-text", no_args_is_help=True)


@app.command("audio-review")
def audio_review(
    data_directory: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
    frontend_directory: Annotated[Path, typer.Option("--frontend-dir")] = Path("audio-review-ui"),
    open_browser: Annotated[bool, typer.Option("--open-browser/--no-open-browser")] = True,
) -> None:
    """Run the local recording and transcript playback workspace."""
    serve_audio_review(
        data_root=data_directory,
        frontend_directory=frontend_directory,
        open_browser=open_browser,
    )


@app.command("transcribe-recordings")
def transcribe_recordings(
    source_directory: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")] = Path("data/rawRecord/转文字文档.md"),
    model_directory: Annotated[Path | None, typer.Option("--model-dir")] = None,
    run_local: Annotated[bool, typer.Option("--local/--no-local")] = False,
    run_cloud: Annotated[bool, typer.Option("--cloud/--no-cloud")] = True,
    allow_cloud_upload: Annotated[
        bool, typer.Option("--allow-cloud-upload/--no-allow-cloud-upload")
    ] = True,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    cost_cap_cny: Annotated[float, typer.Option("--cost-cap-cny", min=0.01)] = 30.0,
    speech_only_cloud: Annotated[bool, typer.Option("--speech-only-cloud/--full-cloud")] = True,
    speech_region_gap_seconds: Annotated[int, typer.Option("--speech-region-gap-seconds", min=0)] = 30,
    third_pass: Annotated[bool, typer.Option("--third-pass/--no-third-pass")] = False,
    deepseek_text: Annotated[bool, typer.Option("--deepseek-text/--no-deepseek-text")] = False,
    deepseek_cost_cap_cny: Annotated[float, typer.Option("--deepseek-cost-cap-cny", min=0.01)] = 0.6,
    expected_meetings: Annotated[int | None, typer.Option("--expected-meetings", min=1)] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Transcribe long recordings with local/cloud evidence and speaker memory."""
    try:
        from research_kb.audio import pipeline

        settings = AppSettings()
        api_key = settings.qwen_api_key.get_secret_value() if settings.qwen_api_key else None
        deepseek_api_key = (
            settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else None
        )
        document = pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(
                source_directory=source_directory,
                output_path=output,
                model_directory=model_directory or settings.audio_model_dir,
                run_local=run_local,
                run_cloud=run_cloud,
                allow_cloud_upload=allow_cloud_upload,
                resume=resume,
                cost_cap_cny=cost_cap_cny,
                speech_only_cloud=speech_only_cloud,
                speech_region_gap_seconds=speech_region_gap_seconds,
                run_third_pass=third_pass,
                run_deepseek_text=deepseek_text,
                deepseek_cost_cap_cny=deepseek_cost_cap_cny,
                expected_meetings=expected_meetings,
            ),
            api_key=api_key,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=settings.deepseek_base_url,
            deepseek_model=settings.deepseek_model,
            deepseek_timeout_seconds=settings.deepseek_timeout_seconds,
        ).run()
    except (OSError, ValidationError, pipeline.RecordingTranscriptionError) as error:
        payload = {"error": str(error), "status": "error"}
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False))
        else:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=2) from error
    payload = {
        "status": "ok",
        "sources": len(document.sources),
        "segments": len(document.segments),
        "speakers": len(document.speakers),
        "output": str(output.expanduser().resolve()),
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")


if __name__ == "__main__":
    app()
