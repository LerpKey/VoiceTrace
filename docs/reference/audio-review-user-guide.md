# VoiceTrace Audio Review User Guide

[English](audio-review-user-guide.md) · [Chinese](audio-review-user-guide.zh-CN.md)

## 1. Release status

Version `0.1.0` is suitable for personal development use and testing on a Windows machine.

| Use case | Result | Reason |
|---|---|---|
| Personal use on the current machine | Supported | Playback, transcript reading, topic navigation, speaker display names, favorites, notes, and logs are implemented |
| Multiple people on the same LAN | Not supported | The service listens on the local machine only and has no account or concurrent-editing design |
| Public internet deployment | Do not publish directly | Audio and state remain local; authentication, remote storage, and server-side privacy isolation are not implemented |

In this guide, “release” means delivery of the local application. It does not mean publishing private recordings to the internet.

## 2. Purpose

The recording review workspace is intended to:

- play long recordings and jump to relevant moments;
- compare transcript sentences with the original audio;
- browse by wall-clock time and recording-relative time;
- review the complete topic flow;
- set a consistent display name for a speaker across a recording;
- favorite important sentences and add notes in one place;
- add recordings and start transcription.

The workspace is not an automatic finalization tool. For passages marked `[Uncertain: …]`, `[Unclear]`, or containing overlapping speech, replay the original audio before using the text in a formal document.

## 3. Start and stop

### 3.1 Daily start

The simplest start method is to double-click `start-voicetrace-workspace.bat` in the project root. The launcher starts the local service and opens `http://localhost:3000/`. If the workspace is already running, it only opens the page again.

Keep the launcher window open while using the workspace. Press `Ctrl+C` in that window to stop it.

To start it manually from PowerShell:

```powershell
cd "project root"
uv run voice-trace audio-review --data-dir data
```

The service opens:

```text
http://localhost:3000/
```

`--open-browser` is enabled by default. Use `--no-open-browser` to start the service without opening a browser:

```powershell
uv run voice-trace audio-review --data-dir data --no-open-browser
```

If the frontend is in another directory:

```powershell
uv run voice-trace audio-review --data-dir data `
  --frontend-dir audio-review-ui
```

Run `uv run voice-trace audio-review --help` for the current options.

### 3.2 First install or a new machine

The project uses Python 3.12 and uv:

```powershell
uv sync
```

Install the frontend from its lockfile:

```powershell
cd "project root\audio-review-ui"
npm ci
```

Do not mix npm and pnpm in the same frontend directory.

### 3.3 Stop

Return to the PowerShell window that started the application and press `Ctrl+C`. This stops the local page and API together.

## 4. Workspace layout

The page has three main areas:

1. Top left: player, wall-clock time, recording-relative progress, and the full-day timeline;
2. Bottom left: the transcript for the current 30-minute window;
3. Right: the Summary view, or the current recording's favorites and permanent activity log.

The top bar opens the recording library, recording manager, favorites summary, and upload panel. The recording library is grouped by month and sorted by current display name. Search matches the current name and original filename.

Use the language selector in the top bar or favorites page to switch between English and Chinese. The choice is stored in browser `localStorage` and is shared by both pages and tabs.

## 5. Playback and navigation

### 5.1 Basic controls

- Play or pause with the player button;
- Use `−10` and `+10` to seek by ten seconds;
- Choose 0.75×, 1×, 1.25×, 1.5×, or 2× playback speed;
- Click anywhere on the full-day timeline to seek;
- Click a transcript sentence to play its original audio;
- The sentence at the current playback position is highlighted automatically.

### 5.2 Keyboard shortcuts

| Key | Action |
|---|---|
| Space | Play or pause |
| Left arrow | Seek back 5 seconds |
| Right arrow | Seek forward 5 seconds |
| `Shift` + Left arrow | Seek back 30 seconds |
| `Shift` + Right arrow | Seek forward 30 seconds |

Shortcuts do not take over while typing in a name, search field, or note.

### 5.3 Follow playback

While playing, the current sentence scrolls to the center of the transcript. Manual scrolling pauses this behavior. Click `Return to playback` to resume following.

## 6. Time display and calibration

Each transcript block shows:

- **Actual**: the wall-clock time calculated from the recording start time;
- **In recording**: the relative offset from the start of the file.

If the recording was edited or its file time is unreliable:

1. Click `Calibrate start time`;
2. Enter the correct date and start time;
3. Click `Save calibration`.

Calibration changes only the wall-clock conversion. It does not change the audio, duration, or original transcript. Click `Restore file time` to undo it.

## 7. Transcript and topics

### 7.1 Transcript

The transcript merges adjacent sentences with continuous meaning while keeping independent time anchors and speaker labels. The gray background marks uncertain passages; replay the original audio to verify them.

`View original recognition and model candidates` exposes the raw recognition candidates. The reviewed text is intended to improve punctuation, sentence breaks, and high-confidence errors only. It must not turn an `[Uncertain: …]` passage into confirmed evidence.

### 7.2 Summary

The Summary view uses a prompt to select and organize topics from the complete transcript. Prompt templates are written in English and explicitly tell the model to keep the source language of the recording and transcript; user-provided prompts may request another scope, but the audio content is not silently translated.

The prompt library, progress bar, failure reasons, retry behavior, and export actions are available from the right panel. Summary generation does not overwrite the original transcript.

## 8. Manage speakers

1. Select a recording;
2. Click `Manage speakers`;
3. Enter the desired display name beside an original speaker label;
4. Click `Save`.

The change applies to that original label throughout the recording. Multiple original labels may share a display name. Click `Restore` to return to the original label.

This changes the display layer only; it does not change speaker clustering or `transcript.json`. If two people were incorrectly grouped under one label, renaming cannot separate them.

## 9. Favorites, summary, and notes

### 9.1 Favorite a sentence

Click `☆` beside a sentence. It becomes `★`, receives a yellow highlight, and appears in the current recording's right panel. Click it again to remove the favorite. The activity log remains append-only.

### 9.2 Favorites summary

Click `Favorites summary` to open all recordings' favorites in a new tab:

```text
http://localhost:3000/favorites
```

The page groups favorites by recording and searches recording names, speakers, sentence text, and notes. The workspace notifies the summary page immediately after a star change. The page also refreshes when it becomes visible, receives focus, every five seconds, or when `Refresh favorites` is clicked.

### 9.3 Add a note

1. Click the note area beside a favorite;
2. Enter up to 2,000 characters;
3. Click `Save note`.

Long notes are ellipsized in the preview. Hover to read the full note, or open it on a touch device. Note changes and clearing are written to the permanent log. Removing a favorite removes the current note from the favorite record, but not from the log.

## 10. Manage recording names

The current recording name appears above the player. Click it to edit, then press `Enter` or click outside to save; press `Escape` to cancel. This changes the workspace display name only and never renames the original audio file.

`Manage recordings` can search by display name or original filename, save a display name, restore the original name, hide a recording, and restore a hidden recording. Hiding is reversible and does not delete audio or transcripts.

## 11. Add a recording

1. Click `＋ Add recording`;
2. Choose M4A, MP3, WAV, or FLAC, up to 2 GB per file;
3. Keep `Low-cost cloud enhancement` enabled for the default speech-only cloud path, or turn it off for local-only processing;
4. Choose the file and start transcription.

Cloud enhancement means that only VAD-selected speech intervals are uploaded to the configured recognition service. The hard cap is ¥3 per task. Do not enable it for recordings containing private, meeting, or personal information without explicit authorization.

The first formal speech chunk is also used for cloud model selection; no separate probe recording is uploaded. When an old task is resumed, an existing `pilot.json` counts as historical usage and is not requested again. The cost ledger deduplicates provider task IDs. ASR prices are estimates based on the provider's published list price; credits, promotions, and the final bill are controlled by the provider console.

Tasks use file hashes and manifests for checkpointed execution. Multiple recordings run one at a time. Temporary failures retry up to three times, and unfinished tasks resume after a service restart. Matching files and cloud settings reuse successful caches. Configuration errors are shown directly; a DeepSeek failure keeps the ASR transcript. A completed task can repair text review without re-uploading audio. If the text sub-budget is exhausted, keep the original text or add budget within the task cap. When a task cannot continue automatically, the task card provides stage-specific continue and cancel choices.

`Clear cache and start over` removes only the current task's transcription directory, model cache, and manifest. It keeps the uploaded original audio. Existing cloud charges are not reversed, and new external cost remains subject to the task cap. If the original audio is damaged, restarting reports the same file error.

After upload, the selected local/cloud ASR path runs. Enabling cloud enhancement with initial text review also runs one initial DeepSeek text-review request. Clicking `Summary` later creates a separate DeepSeek summary request. Re-running a summary writes `summary_result.json` and `topic-index.md`; it does not overwrite `text_analysis.json` or `transcript.json`, and it does not change the transcription job's completed stages.

The `Transcription jobs` panel refreshes every two seconds and persists across page refreshes. It shows the current stage, completion percentage, DeepSeek input/output tokens and cost, cloud ASR billed seconds and cost, and estimated total cost. ASR is billed by effective audio duration and does not consume text tokens. When cloud enhancement is disabled, ASR and DeepSeek cost are both ¥0. In-progress values are estimates from provider usage; the completed manifest is authoritative.

The task card explains the recovery strategy and maximum additional external cost before a continue action. Missing, damaged, or resource-constrained recordings do not receive an unsafe continue button. Queued or running tasks can be cancelled; submitted cloud usage may still be billed. Failed, completed, and cancelled tasks can be removed from the list without deleting the upload, manifest, or existing transcript artifacts.

If a removed failed task is uploaded again, it will not block the new task. An unremoved failed task can still reuse its checkpoint by file hash. The error `speech-only cloud mode requires cloud ASR` means the task does not have cloud ASR enabled. Speech-only cloud mode uses local VAD and speaker models for interval selection and speaker association; it does not depend on local Qwen3-ASR.

## 12. Data, security, and backup

The workspace listens on the local machine only. It does not intentionally expose the page or audio to the LAN or public internet. Original audio and raw transcripts are not overwritten by renaming, speaker display names, time calibration, favorites, or notes.

Human-edited state is stored in:

```text
data/audio-review/
├─ recording-time-overrides.json
├─ recording-catalog.json
├─ speaker-overrides.json
├─ favorites.json
└─ activity-log.jsonl
```

Back up the complete `data/audio-review/` directory to preserve manual changes. For a full workspace migration, also back up the audio, `transcript.json`, `reading_view.json`, analysis results, and manifests under `data/`.

Do not manually edit JSON or JSONL files while the workspace is running. Stop the workspace before restoring a backup.

## 13. Troubleshooting

### Blank or unreachable page

1. Press `Ctrl+F5`;
2. If the page is still unavailable, close the launcher window and run the daily start command again;
3. Confirm that only one workspace is running;
4. Do not mix npm and pnpm in the same frontend dependency directory.

### The launcher says that `node` is not recognized

The current launcher supplies the project-local Node runtime to the frontend. Press `Ctrl+C` in the launcher window and double-click `start-voicetrace-workspace.bat` again. Keep the full startup output if the problem persists.

### The local recording service is unavailable

The API is not running or its port is occupied. Stop the old process and run:

```powershell
uv run voice-trace audio-review --data-dir data
```

### Favorites do not update immediately

Wait up to five seconds, return to the favorites tab, or click `Refresh favorites`. If the pages still disagree, refresh both tabs and confirm that the local API is running.

### Wall-clock time is wrong

Use `Calibrate start time`. Recording-relative time is always based on the file and should not be corrected by editing the audio.

### Speaker labels are wrong

If only the name is wrong, use `Manage speakers`. If one label contains several real people, this is a speaker-separation issue; keep the uncertainty instead of trying to fix it with a display name.

### The transcript is inaccurate

Replay the original audio and expand the raw recognition candidates. Confirm numbers, names, addresses, and proper nouns manually before formal use.

## 14. Current limitations and public-release prerequisites

The local version does not edit transcript prose or export a formal document. Speaker management changes display names only, and favorite notes are not collaborative editing.

Public or team deployment would at least require:

1. User login, access control, and operation ownership;
2. Encrypted remote audio storage with lifecycle policies;
3. A server-side database, backups, and concurrent-edit conflict handling;
4. HTTPS, audit permissions, and privacy notices;
5. An installer, automatic updates, failure recovery, and real multi-user acceptance testing;
6. A clear compliance and cost policy for cloud transcription data.

Until these conditions are met, do not expose the local API to the LAN or public internet.
