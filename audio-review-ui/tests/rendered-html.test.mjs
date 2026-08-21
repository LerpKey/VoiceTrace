import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("ships the recording workspace instead of the starter preview", async () => {
  const [page, layout, css, locale] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
    readFile(new URL("app/locale.tsx", root), "utf8"),
  ]);
  assert.match(page, /VoiceTrace/);
  assert.match(page, /currentTime/);
  assert.match(page, /scrollIntoView/);
  assert.match(page, /api\/recordings/);
  assert.match(page, /summary/);
  assert.match(layout, /lang="en"/);
  assert.match(locale, /languageEnglish/);
  assert.match(locale, /languageChinese/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.match(css, /grid-template-columns/);
  assert.match(css, /@media \(max-width:900px\)/);
});

test("keeps transcript content read-only while exposing source candidates", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /originalCandidates/);
  assert.doesNotMatch(page, /contentEditable|PATCH/);
  assert.match(page, /summary-prompt-input/);
});

test("shows inferred wall-clock time before recording elapsed time", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  const actualTime = page.indexOf('className="actual-block-time"');
  const elapsedTime = page.indexOf('className="elapsed-block-time"');
  assert.ok(actualTime >= 0, "missing inferred wall-clock time");
  assert.ok(elapsedTime > actualTime, "wall-clock time must appear before elapsed time");
  assert.match(page, /actualRange\(selected\?\.recorded_at/);
  assert.match(page, /crossesDay/);
});

test("offers a persistent recording start-time calibration", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /calibrateStart/);
  assert.match(page, /type="datetime-local"/);
  assert.match(page, /method: "PUT"/);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /start_time_overridden/);
});

test("shows a complete topic timeline with strong and weak speech sections", async () => {
  const [page, css, locale] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
    readFile(new URL("app/locale.tsx", root), "utf8"),
  ]);
  assert.match(page, /summaryResult/);
  assert.match(page, /topic_segment_count/);
  assert.match(page, /topic\.segment_count/);
  assert.match(page, /strongTopic/);
  assert.match(page, /weakTopic/);
  assert.match(page, /keepByPrompt/);
  assert.match(page, /omittedOnLeft/);
  assert.match(page, /summary-open-button/);
  assert.match(locale, /deepseekSummary/);
  assert.match(locale, /Finance livestream/);
  assert.match(page, /summary-prompt-add/);
  assert.match(page, /api\/summary-prompts/);
  assert.match(page, /savePrompt/);
  assert.match(page, /summary-progress/);
  assert.match(page, /retrySummary/);
  assert.match(page, /exportSummary/);
  assert.match(page, /exportTranscript/);
  assert.match(page, /export\/\$\{kind/);
  assert.match(css, /export-button/);
  assert.match(css, /topic-card\.weak/);
  assert.match(css, /topic-band\.topic-weak/);
  assert.match(page, /className="topic-summary" title=\{topic\.summary\}/);
  assert.match(page, /title=\{topic\.title\}/);
  assert.match(page, /title=\{word\}/);
  assert.match(css, /topic-card:hover \.topic-summary/);
  assert.match(css, /topic-card:focus-visible \.topic-summary/);
  assert.match(css, /-webkit-line-clamp:2/);
});

test("supports recording-wide speaker names and persistent favorite highlights", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /manageSpeakers/);
  assert.match(page, /speakerInstruction/);
  assert.match(page, /speaker-overrides/);
  assert.match(page, /favorites\/\$\{sentence\.id\}/);
  assert.match(page, /savedAndLogs/);
  assert.match(page, /activity/);
  assert.match(css, /sentence\.favorite/);
  assert.match(css, /speaker-editor/);
  assert.match(css, /favorite-card/);
});

test("manages recording names with search and reversible removal", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /manageRecordingTitle/);
  assert.match(page, /searchRecordingName/);
  assert.match(page, /include_hidden: "true"/);
  assert.match(page, /URLSearchParams/);
  assert.match(page, /readingCatalog/);
  assert.match(page, /\/title/);
  assert.match(page, /\/restore/);
  assert.match(page, /removeDoesNotDelete/);
  assert.match(css, /recording-manager/);
  assert.match(css, /recording-manage-row/);
});

test("selects recordings from filename-sorted month folders and renames inline", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /selectRecording/);
  assert.match(page, /recordingMonthKey/);
  assert.match(page, /localeCompare\(.*numeric: true/);
  assert.match(page, /browseMonths/);
  assert.match(page, /className="recording-month-tabs"/);
  assert.match(page, /className="recording-choice-list"/);
  assert.match(page, /editCurrentTitle/);
  assert.match(page, /onBlur=\{\(\) => void saveInlineRecordingTitle\(\)\}/);
  assert.match(page, /event\.key === "Enter"/);
  assert.match(page, /event\.key === "Escape"/);
  assert.match(css, /recording-browser/);
  assert.match(css, /current-recording-title/);
  assert.match(css, /recording-month-tab/);
  assert.match(css, /recording-choice/);
});

test("summarizes favorites across recordings with persistent hover notes", async () => {
  const [workspace, favorites, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/favorites/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(workspace, /href="\/favorites"/);
  assert.match(workspace, /target="_blank"/);
  assert.match(favorites, /api\/favorites/);
  assert.match(favorites, /favoritesTitle/);
  assert.match(favorites, /favorites\/\$\{favorite\.segment_id\}\/note/);
  assert.match(favorites, /method: "PUT"/);
  assert.match(favorites, /title=\{favorite\.note \|\| t\("addNote"\)\}/);
  assert.match(favorites, /maxLength=\{2_000\}/);
  assert.match(favorites, /BroadcastChannel/);
  assert.match(favorites, /visibilitychange/);
  assert.match(favorites, /setInterval/);
  assert.match(favorites, /refreshFavorites/);
  assert.match(favorites, /LanguageSwitcher/);
  assert.match(css, /favorite-note-preview/);
  assert.match(css, /text-overflow:ellipsis/);
  assert.match(css, /favorite-summary-card/);
});

test("keeps uploaded transcription progress, tokens, and costs visible", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /api\/jobs/);
  assert.match(page, /setInterval/);
  assert.match(page, /transcriptionJobs/);
  assert.match(page, /collapseJobs/);
  assert.match(page, /expandJobs/);
  assert.match(page, /aria-expanded=\{!jobsCollapsed\}/);
  assert.match(page, /transcription-job-list/);
  assert.match(page, /activeJobs\.length/);
  assert.match(page, /processingProgress/);
  assert.match(page, /DeepSeek Token/);
  assert.match(page, /tokenCost/);
  assert.match(page, /asrBilling/);
  assert.match(page, /estimatedTotal/);
  assert.match(page, /job\.warning/);
  assert.match(page, /recordingOptionLabel/);
  assert.match(page, /originalFile/);
  assert.match(page, /refreshCatalog/);
  assert.match(page, /visibilitychange/);
  assert.match(page, /window\.addEventListener\("focus"/);
  assert.match(page, /coreReadyPending/);
  assert.match(page, /coreReadyPartial/);
  assert.match(page, /fullTranscriptReady/);
  assert.match(css, /job-notice/);
  assert.match(page, /attempt_count/);
  assert.match(page, /checkpointRetried/);
  assert.match(page, /decisionRequired/);
  assert.match(page, /continueAsRecommended/);
  assert.match(page, /repairTextOnly/);
  assert.match(page, /localizedRecoveryDecision/);
  assert.match(page, /continue_cloud_with_higher_cap/);
  assert.match(page, /restart_from_scratch/);
  assert.match(page, /decision-restart/);
  assert.match(page, /jobRecoveryOptions/);
  assert.match(page, /JSON\.stringify\(\{ strategy: decision\.strategy \}\)/);
  assert.match(page, /keepCurrentResult/);
  assert.match(page, /cancelKeepExisting/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/continue/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/decision\/cancel/);
  assert.match(page, /cancelTranscription/);
  assert.match(page, /removeRecord/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/cancel/);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /cancelLocalTitle/);
  assert.doesNotMatch(page, /window\.confirm/);
  assert.match(page, /role="progressbar"/);
  assert.match(css, /job-progress/);
  assert.match(css, /job-board-toggle/);
  assert.match(css, /job-cost-grid/);
  assert.match(css, /job-actions/);
  assert.match(css, /job-decision/);
  assert.match(css, /decision-cloud/);
  assert.match(page, /LanguageSwitcher/);
});
