import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("ships the recording workspace instead of the starter preview", async () => {
  const [page, layout, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /录音文本工作台/);
  assert.match(page, /currentTime/);
  assert.match(page, /scrollIntoView/);
  assert.match(page, /api\/recordings/);
  assert.match(page, /总结/);
  assert.match(layout, /lang="zh-CN"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.match(css, /grid-template-columns/);
  assert.match(css, /@media \(max-width:900px\)/);
});

test("keeps transcript content read-only while exposing source candidates", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /查看原始识别与模型候选/);
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
  assert.match(page, /校准开始时间/);
  assert.match(page, /type="datetime-local"/);
  assert.match(page, /method: "PUT"/);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /start_time_overridden/);
});

test("shows a complete topic timeline with strong and weak speech sections", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /总结结果/);
  assert.match(page, /已覆盖.*topic_segment_count/);
  assert.match(page, /topic\.segment_count/);
  assert.match(page, /明确话题/);
  assert.match(page, /零散 \/ 弱话题/);
  assert.match(page, /按提示词保留/);
  assert.match(page, /未纳入总结的内容仍在左侧转写/);
  assert.match(page, /summary-open-button/);
  assert.match(page, /DEEPSEEK SUMMARY/);
  assert.match(page, /财经直播/);
  assert.match(page, /summary-prompt-add/);
  assert.match(page, /api\/summary-prompts/);
  assert.match(page, /保存已修改的提示词/);
  assert.match(page, /summary-progress/);
  assert.match(page, /重新尝试/);
  assert.match(page, /导出总结 MD/);
  assert.match(page, /导出对话 MD/);
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
  assert.match(page, /管理说话人/);
  assert.match(page, /完整录音内统一替换/);
  assert.match(page, /speaker-overrides/);
  assert.match(page, /favorites\/\$\{sentence\.id\}/);
  assert.match(page, /收藏与日志/);
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
  assert.match(page, /管理录音/);
  assert.match(page, /搜索录音名称/);
  assert.match(page, /include_hidden: "true"/);
  assert.match(page, /URLSearchParams/);
  assert.match(page, /正在读取录音目录/);
  assert.match(page, /\/title/);
  assert.match(page, /\/restore/);
  assert.match(page, /只从工作台移除，不删除原始音频和转写/);
  assert.match(css, /recording-manager/);
  assert.match(css, /recording-manage-row/);
});

test("selects recordings from filename-sorted month folders and renames inline", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /选择录音/);
  assert.match(page, /recordingMonthKey/);
  assert.match(page, /localeCompare\(.*numeric: true/);
  assert.match(page, /按月份浏览/);
  assert.match(page, /className="recording-month-tabs"/);
  assert.match(page, /className="recording-choice-list"/);
  assert.match(page, /单击修改录音名称/);
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
  assert.match(favorites, /收藏语句汇总/);
  assert.match(favorites, /favorites\/\$\{favorite\.segment_id\}\/note/);
  assert.match(favorites, /method: "PUT"/);
  assert.match(favorites, /title=\{favorite\.note \|\| "点击添加备注"\}/);
  assert.match(favorites, /maxLength=\{2_000\}/);
  assert.match(favorites, /BroadcastChannel/);
  assert.match(favorites, /visibilitychange/);
  assert.match(favorites, /setInterval/);
  assert.match(favorites, /刷新收藏/);
  assert.doesNotMatch(favorites, /localStorage|sessionStorage/);
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
  assert.match(page, /转写任务/);
  assert.match(page, /收起转写任务/);
  assert.match(page, /展开转写任务/);
  assert.match(page, /aria-expanded=\{!jobsCollapsed\}/);
  assert.match(page, /transcription-job-list/);
  assert.match(page, /activeJobs\.length/);
  assert.match(page, /处理进度/);
  assert.match(page, /DeepSeek Token/);
  assert.match(page, /Token 费用/);
  assert.match(page, /ASR 计费/);
  assert.match(page, /总费用/);
  assert.match(page, /job\.warning/);
  assert.match(page, /recordingOptionLabel/);
  assert.match(page, /原文件：/);
  assert.match(page, /refreshCatalog/);
  assert.match(page, /visibilitychange/);
  assert.match(page, /window\.addEventListener\("focus"/);
  assert.match(page, /核心转写已完成/);
  assert.match(page, /文本整理部分完成/);
  assert.match(page, /完整 ASR 转写已生成/);
  assert.match(css, /job-notice/);
  assert.match(page, /attempt_count/);
  assert.match(page, /自动断点重试/);
  assert.match(page, /需要你的决定/);
  assert.match(page, /按建议继续/);
  assert.match(page, /仅修复文本整理/);
  assert.match(page, /追加预算完成剩余窗口/);
  assert.match(page, /continue_cloud_with_higher_cap/);
  assert.match(page, /restart_from_scratch/);
  assert.match(page, /decision-restart/);
  assert.match(page, /jobRecoveryOptions/);
  assert.match(page, /JSON\.stringify\(\{ strategy: decision\.strategy \}\)/);
  assert.match(page, /暂不修复，保留当前结果/);
  assert.match(page, /取消并保留已有结果/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/continue/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/decision\/cancel/);
  assert.match(page, /取消转写/);
  assert.match(page, /移除记录/);
  assert.match(page, /api\/jobs\/\$\{job\.id\}\/cancel/);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /已经产生的云端费用仍可能计费/);
  assert.doesNotMatch(page, /window\.confirm/);
  assert.match(page, /role="progressbar"/);
  assert.match(css, /job-progress/);
  assert.match(css, /job-board-toggle/);
  assert.match(css, /job-cost-grid/);
  assert.match(css, /job-actions/);
  assert.match(css, /job-decision/);
  assert.match(css, /decision-cloud/);
  assert.doesNotMatch(page, /localStorage|sessionStorage/);
});

