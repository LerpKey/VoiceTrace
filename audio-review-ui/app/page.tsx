"use client";

import Link from "next/link";
import { ChangeEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { announceFavoriteChange } from "./favorite-events";

const API = process.env.NEXT_PUBLIC_AUDIO_API ?? "http://127.0.0.1:8765";
const WINDOW_MS = 30 * 60_000;

type Recording = {
  id: string;
  title: string;
  original_title?: string;
  title_overridden?: boolean;
  hidden?: boolean;
  duration_ms: number;
  recorded_at: string;
  original_recorded_at?: string;
  start_time_overridden?: boolean;
  status: string;
  segment_count: number;
  favorite_count?: number;
  has_transcript: boolean;
  has_topics: boolean;
};

type Candidate = { provider: string; model: string; text: string };
type SummaryPrompt = {
  id: string;
  name: string;
  prompt: string;
  created_at?: string;
  updated_at?: string;
};
type Sentence = {
  id: string;
  start_ms: number;
  end_ms: number;
  speaker: string;
  original_speaker: string;
  text: string;
  confidence: number;
  flags: string[];
  candidates: Candidate[];
};
type TextBlock = { id: string; start_ms: number; end_ms: number; sentences: Sentence[] };
type Topic = {
  id: string;
  title: string;
  summary: string;
  keywords: string[];
  evidence_segment_ids: string[];
  start_ms: number;
  end_ms: number;
  segment_count: number;
  strength: "strong" | "weak";
};
type Detail = Recording & {
  density: number[];
  topics: Topic[];
  topic_segment_count: number;
  processing?: {
    status: string;
    completed_steps: string[];
    estimated_cost_cny: number;
    cost_cap_cny: number | null;
    errors: string[];
  };
  summary?: {
    status: "idle" | "queued" | "running" | "completed" | "failed" | "unavailable";
    source?: "none" | "text_review" | "summary";
    prompt?: string;
    error?: string | null;
    progress_percent?: number;
    stage?: string;
    attempt_count?: number;
    max_attempts?: number;
    retry_count?: number;
    can_retry?: boolean;
    cost_cny?: number | null;
    topic_count?: number;
  };
};
type SpeakerInfo = {
  source_speaker: string;
  display_name: string;
  segment_count: number;
  is_overridden: boolean;
};
type Favorite = {
  segment_id: string;
  start_ms: number;
  end_ms: number;
  speaker: string;
  text: string;
  created_at: string;
  note?: string;
  note_updated_at?: string | null;
};
type ActivityEvent = {
  event_id: string;
  created_at: string;
  action: string;
  details: Record<string, unknown>;
};
type RecoveryDecision = {
  strategy: string;
  title: string;
  description: string;
  impact: string;
  continue_label: string;
  can_continue: boolean;
  additional_external_cost_cny: number;
  cost_cap_cny?: number;
};
type TranscriptionJob = {
  id: string;
  filename: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  progress_percent: number;
  allow_cloud_upload: boolean;
  completed_steps: string[];
  cloud_billed_seconds: number;
  asr_cost_cny: number;
  text_review_input_tokens: number;
  text_review_output_tokens: number;
  text_review_total_tokens: number;
  text_review_cost_cny: number;
  text_review_cost_cap_cny: number;
  estimated_cost_cny: number;
  cost_cap_cny: number;
  attempt_count?: number;
  max_attempts?: number;
  retry_count?: number;
  recovery_count?: number;
  cancel_requested?: boolean;
  retrying?: boolean;
  last_error?: string | null;
  error?: string | null;
  warning?: string | null;
  core_transcript_ready?: boolean;
  text_review_status?: "not_requested" | "running" | "completed" | "partial" | "fallback" | "failed";
  recovery_decision?: RecoveryDecision | null;
  recovery_options?: RecoveryDecision[];
};

const SUMMARY_EXAMPLES: SummaryPrompt[] = [
  {
    id: "prompt-complete-meeting",
    name: "完整会议",
    prompt: "请按录音时间顺序完整梳理会议内容，不遗漏议题、决定、分工、数字和待办；保留不确定信息，不要擅自补全。",
  },
  {
    id: "prompt-finance-live",
    name: "财经直播",
    prompt: "只整理主播的财经相关内容：市场观点、个股/行业、宏观数据、风险提示和操作逻辑。删除感谢礼物、寒暄、唱歌、闲聊和其他非财经内容，不要把这些内容列入总结。",
  },
];

const UNCERTAIN_FLAGS = new Set(["models_disagree", "no_majority", "sensitive_difference", "vad_supplement"]);

function isUncertainSentence(sentence: Sentence) {
  return sentence.text.startsWith("[疑似：") || sentence.flags.some((flag) => UNCERTAIN_FLAGS.has(flag));
}

function recordingOptionLabel(recording: Recording) {
  if (
    recording.title_overridden &&
    recording.original_title &&
    recording.original_title !== recording.title
  ) {
    return `${recording.title}（原文件：${recording.original_title}）`;
  }
  return recording.title;
}

function recordingMonthKey(recordedAt: string) {
  const date = new Date(recordedAt);
  if (Number.isNaN(date.getTime())) return "unknown";
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

function recordingMonthLabel(monthKey: string) {
  if (monthKey === "unknown") return "日期未确定";
  const [year, month] = monthKey.split("-");
  return `${year}年${Number(month)}月`;
}

function sortRecordingsByTitle(left: Recording, right: Recording) {
  return left.title.localeCompare(right.title, "zh-CN", { numeric: true, sensitivity: "base" });
}

function jobCompletionSummary(job: TranscriptionJob) {
  if (job.core_transcript_ready && job.text_review_status === "fallback") {
    return `核心转写已完成 · 文本整理待处理 · ${job.completed_steps.length} 个阶段已完成`;
  }
  if (job.core_transcript_ready && job.text_review_status === "partial") {
    return `核心转写已完成 · 文本整理部分完成 · ${job.completed_steps.length} 个阶段已完成`;
  }
  return `${job.completed_steps.length} 个阶段已完成`;
}

function jobRecoveryOptions(job: TranscriptionJob) {
  if (job.recovery_options?.length) return job.recovery_options;
  return job.recovery_decision ? [job.recovery_decision] : [];
}

function clock(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h, m, s].map((part) => String(part).padStart(2, "0")).join(":");
}

function actualClock(recordedAt: string, ms: number) {
  const date = new Date(new Date(recordedAt).getTime() + ms);
  return Number.isNaN(date.getTime())
    ? clock(ms)
    : new Intl.DateTimeFormat("zh-CN", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(date);
}

function actualRange(recordedAt: string, startMs: number, endMs: number) {
  const recordedTime = new Date(recordedAt).getTime();
  if (Number.isNaN(recordedTime)) return "实际时间未知";

  const start = new Date(recordedTime + startMs);
  const end = new Date(recordedTime + endMs);
  const crossesDay =
    start.getFullYear() !== end.getFullYear() ||
    start.getMonth() !== end.getMonth() ||
    start.getDate() !== end.getDate();
  const formatter = new Intl.DateTimeFormat("zh-CN", {
    ...(crossesDay ? { month: "2-digit", day: "2-digit" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  return `${formatter.format(start)} 至 ${formatter.format(end)}`;
}

function dateTimeInputValue(recordedAt: string) {
  const date = new Date(recordedAt);
  if (Number.isNaN(date.getTime())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 19);
}

function recordingOffset(recordedAt: string) {
  return recordedAt.match(/(Z|[+-]\d{2}:\d{2})$/)?.[1] ?? "+08:00";
}

function statusText(recording: Recording) {
  if (recording.status === "completed") return `${recording.segment_count.toLocaleString()} 条语音`;
  if (recording.status === "available_with_warning") return `${recording.segment_count.toLocaleString()} 条语音 · 已本地回退整理`;
  if (recording.status === "failed") return "处理遇到问题";
  return "转写处理中";
}

function activityText(event: ActivityEvent) {
  const details = event.details;
  if (event.action === "favorite_added") return `收藏了 ${clock(Number(details.start_ms ?? 0))} 的语句`;
  if (event.action === "favorite_removed") return `取消收藏 ${clock(Number(details.start_ms ?? 0))} 的语句`;
  if (event.action === "favorite_note_updated") return "更新了收藏语句的备注";
  if (event.action === "favorite_note_cleared") return "清空了收藏语句的备注";
  if (event.action === "speaker_override_removed") return `恢复“${String(details.source_speaker ?? "说话人")}”的原始名称`;
  if (event.action === "speaker_override_updated") {
    return `将“${String(details.source_speaker ?? "说话人")}”全局显示为“${String(details.display_name ?? "")}”`;
  }
  if (event.action === "recording_title_updated") return `录音改名为“${String(details.title ?? "")}”`;
  if (event.action === "recording_title_reset") return `恢复录音原名“${String(details.title ?? "")}”`;
  if (event.action === "recording_hidden") return `从工作台移除录音“${String(details.title ?? "")}”`;
  if (event.action === "recording_restored") return `恢复录音“${String(details.title ?? "")}”`;
  return event.action;
}

export default function Home() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);
  const [blocks, setBlocks] = useState<TextBlock[]>([]);
  const [currentMs, setCurrentMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [follow, setFollow] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  // The normal product path is low-cost cloud enhancement: VAD filters out
  // silence/noise first, then only speech intervals are uploaded.
  const [allowCloud, setAllowCloud] = useState(true);
  const [jobs, setJobs] = useState<TranscriptionJob[]>([]);
  const [jobsCollapsed, setJobsCollapsed] = useState(false);
  const [jobAction, setJobAction] = useState("");
  const [showStartTimeEditor, setShowStartTimeEditor] = useState(false);
  const [startTimeDraft, setStartTimeDraft] = useState("");
  const [savingStartTime, setSavingStartTime] = useState(false);
  const [startTimeMessage, setStartTimeMessage] = useState("");
  const [showSpeakerEditor, setShowSpeakerEditor] = useState(false);
  const [showRecordingPicker, setShowRecordingPicker] = useState(false);
  const [recordingPickerQuery, setRecordingPickerQuery] = useState("");
  const [activeRecordingMonth, setActiveRecordingMonth] = useState("");
  const [editingCurrentTitle, setEditingCurrentTitle] = useState(false);
  const [currentTitleDraft, setCurrentTitleDraft] = useState("");
  const [currentTitleMessage, setCurrentTitleMessage] = useState("");
  const [showRecordingManager, setShowRecordingManager] = useState(false);
  const [managedRecordings, setManagedRecordings] = useState<Recording[]>([]);
  const [recordingQuery, setRecordingQuery] = useState("");
  const [recordingDrafts, setRecordingDrafts] = useState<Record<string, string>>({});
  const [loadingRecordingCatalog, setLoadingRecordingCatalog] = useState(false);
  const [savingRecording, setSavingRecording] = useState("");
  const [recordingManagerMessage, setRecordingManagerMessage] = useState("");
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>([]);
  const [speakerDrafts, setSpeakerDrafts] = useState<Record<string, string>>({});
  const [savingSpeaker, setSavingSpeaker] = useState("");
  const [speakerMessage, setSpeakerMessage] = useState("");
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [rightView, setRightView] = useState<"timeline" | "favorites">("timeline");
  const [showSummaryDialog, setShowSummaryDialog] = useState(false);
  const [summaryPromptDraft, setSummaryPromptDraft] = useState("");
  const [summaryPrompts, setSummaryPrompts] = useState<SummaryPrompt[]>(SUMMARY_EXAMPLES);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [promptNameDraft, setPromptNameDraft] = useState("");
  const [promptBodyDraft, setPromptBodyDraft] = useState("");
  const [promptSaving, setPromptSaving] = useState(false);
  const [promptLibraryMessage, setPromptLibraryMessage] = useState("");
  const [summarySubmitting, setSummarySubmitting] = useState(false);
  const [summaryMessage, setSummaryMessage] = useState("");
  const [summaryRunning, setSummaryRunning] = useState(false);
  const [exportingMarkdown, setExportingMarkdown] = useState<"summary" | "transcript" | "">("");
  const [exportMessage, setExportMessage] = useState("");
  const [contentRevision, setContentRevision] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const sentenceRefs = useRef(new Map<string, HTMLButtonElement>());
  const programmaticScroll = useRef(false);
  const seenCompletedJobs = useRef(new Set<string>());
  const inlineTitleCancelled = useRef(false);

  const selected = recordings.find((item) => item.id === selectedId) ?? null;
  const activeSentence = useMemo(
    () =>
      blocks.flatMap((block) => block.sentences).find(
        (sentence) => currentMs >= sentence.start_ms && currentMs < sentence.end_ms,
      ),
    [blocks, currentMs],
  );
  const activeTopic = detail?.topics.find(
    (topic) => currentMs >= topic.start_ms && currentMs <= topic.end_ms,
  );
  const favoriteIds = useMemo(
    () => new Set(favorites.map((favorite) => favorite.segment_id)),
    [favorites],
  );
  const activeJobs = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "running"),
    [jobs],
  );
  const visibleJobs = activeJobs.length > 0 ? jobs.slice(0, 5) : jobs.slice(0, 3);
  const recordingGroups = useMemo(() => {
    const query = recordingPickerQuery.trim().toLocaleLowerCase("zh-CN");
    const matches = recordings.filter((recording) =>
      !query || recording.title.toLocaleLowerCase("zh-CN").includes(query)
        || recording.original_title?.toLocaleLowerCase("zh-CN").includes(query),
    );
    const grouped = new Map<string, Recording[]>();
    [...matches].sort(sortRecordingsByTitle).forEach((recording) => {
      const month = recordingMonthKey(recording.recorded_at);
      grouped.set(month, [...(grouped.get(month) ?? []), recording]);
    });
    return [...grouped.entries()]
      .sort(([left], [right]) => {
        if (left === "unknown") return 1;
        if (right === "unknown") return -1;
        return right.localeCompare(left);
      })
      .map(([key, items]) => ({ key, label: recordingMonthLabel(key), recordings: items }));
  }, [recordingPickerQuery, recordings]);
  const activeRecordingGroup = recordingGroups.find((group) => group.key === activeRecordingMonth)
    ?? recordingGroups[0]
    ?? null;

  useEffect(() => {
    if (editingCurrentTitle) return;
    const timer = window.setTimeout(() => setCurrentTitleDraft(selected?.title ?? ""), 0);
    return () => window.clearTimeout(timer);
  }, [editingCurrentTitle, selected?.id, selected?.title]);

  useEffect(() => {
    if (!showRecordingPicker || !recordingGroups.length) return;
    if (!recordingGroups.some((group) => group.key === activeRecordingMonth)) {
      const timer = window.setTimeout(() => setActiveRecordingMonth(recordingGroups[0].key), 0);
      return () => window.clearTimeout(timer);
    }
  }, [activeRecordingMonth, recordingGroups, showRecordingPicker]);

  const loadRecordingExtras = useCallback(async (recordingId: string) => {
    const [speakerResponse, favoriteResponse, activityResponse] = await Promise.all([
      fetch(`${API}/api/recordings/${recordingId}/speakers`),
      fetch(`${API}/api/recordings/${recordingId}/favorites`),
      fetch(`${API}/api/recordings/${recordingId}/activity?limit=100`),
    ]);
    if (!speakerResponse.ok || !favoriteResponse.ok || !activityResponse.ok) {
      throw new Error("无法读取说话人与收藏记录");
    }
    return {
      speakers: (await speakerResponse.json()) as SpeakerInfo[],
      favorites: (await favoriteResponse.json()) as Favorite[],
      activity: (await activityResponse.json()) as ActivityEvent[],
    };
  }, []);

  const loadRecordings = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/recordings`);
      if (!response.ok) throw new Error("本地录音服务暂时不可用");
      const data = (await response.json()) as Recording[];
      setRecordings(data);
      setSelectedId((current) =>
        data.some((recording) => recording.id === current)
          ? current
          : data.find((item) => item.has_transcript)?.id || data[0]?.id || "",
      );
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法读取录音");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadJobs = useCallback(async () => {
    const response = await fetch(`${API}/api/jobs`);
    if (!response.ok) throw new Error("无法读取转写任务进度");
    const data = (await response.json()) as TranscriptionJob[];
    setJobs(data);
    if (data.some((job) => job.status === "queued" || job.status === "running")) {
      setShowUpload(true);
    }
  }, []);

  const loadSummaryPrompts = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/summary-prompts`);
      if (!response.ok) throw new Error("无法读取提示词库");
      const data = (await response.json()) as SummaryPrompt[];
      if (data.length) setSummaryPrompts(data);
    } catch {
      // Keep the two local starter prompts available when the API is temporarily offline.
    }
  }, []);

  const loadManagedRecordings = useCallback(async (query = "") => {
    setLoadingRecordingCatalog(true);
    setRecordingManagerMessage("");
    try {
      const params = new URLSearchParams({ include_hidden: "true" });
      if (query.trim()) params.set("q", query.trim());
      const response = await fetch(`${API}/api/recordings?${params.toString()}`);
      if (!response.ok) throw new Error("无法读取录音目录");
      const data = (await response.json()) as Recording[];
      setManagedRecordings(data);
      setRecordingDrafts(
        Object.fromEntries(data.map((recording) => [recording.id, recording.title])),
      );
    } finally {
      setLoadingRecordingCatalog(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadSummaryPrompts(), 0);
    return () => window.clearTimeout(timer);
  }, [loadSummaryPrompts]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${API}/api/recordings`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("本地录音服务暂时不可用");
        return response.json() as Promise<Recording[]>;
      })
      .then((data) => {
        setRecordings(data);
        setSelectedId(data.find((item) => item.has_transcript)?.id || data[0]?.id || "");
        setError("");
      })
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(caught.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      void loadJobs().catch((caught: Error) => setUploadMessage(caught.message));
    }, 0);
    const timer = window.setInterval(() => {
      void loadJobs().catch(() => undefined);
    }, 2_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [loadJobs]);

  useEffect(() => {
    const completed = new Set(
      jobs.filter((job) => job.status === "completed").map((job) => job.id),
    );
    const hasNewCompletion = [...completed].some((jobId) => !seenCompletedJobs.current.has(jobId));
    seenCompletedJobs.current = completed;
    if (hasNewCompletion) void loadRecordings();
  }, [jobs, loadRecordings]);

  useEffect(() => {
    const refreshCatalog = () => void loadRecordings();
    const refreshVisibleCatalog = () => {
      if (document.visibilityState === "visible") refreshCatalog();
    };
    const timer = window.setInterval(refreshCatalog, 15_000);
    window.addEventListener("focus", refreshCatalog);
    document.addEventListener("visibilitychange", refreshVisibleCatalog);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshCatalog);
      document.removeEventListener("visibilitychange", refreshVisibleCatalog);
    };
  }, [loadRecordings]);

  useEffect(() => {
    if (!selectedId) return;
    void fetch(`${API}/api/recordings/${selectedId}`)
      .then((response) => {
        if (!response.ok) throw new Error("无法读取录音详情");
        return response.json() as Promise<Detail>;
      })
      .then((data) => {
        setDetail(data);
        setSummaryRunning(data.summary?.status === "queued" || data.summary?.status === "running");
      })
      .catch((caught: Error) => setError(caught.message));
    void loadRecordingExtras(selectedId)
      .then((extras) => {
        setSpeakers(extras.speakers);
        setSpeakerDrafts(
          Object.fromEntries(
            extras.speakers.map((speaker) => [speaker.source_speaker, speaker.display_name]),
          ),
        );
        setFavorites(extras.favorites);
        setActivity(extras.activity);
      })
      .catch((caught: Error) => setError(caught.message));
  }, [loadRecordingExtras, selectedId]);

  useEffect(() => {
    if (!summaryRunning || !selectedId) return;
    const refreshSummary = () => {
      void fetch(`${API}/api/recordings/${selectedId}`)
        .then((response) => {
          if (!response.ok) throw new Error("无法读取总结进度");
          return response.json() as Promise<Detail>;
        })
        .then((data) => {
          setDetail(data);
          if (data.summary?.status !== "queued" && data.summary?.status !== "running") {
            setSummaryRunning(false);
            if (data.summary?.status === "failed") {
              setSummaryMessage(data.summary.error || "总结失败，请检查 DeepSeek 配置");
            } else {
              setSummaryMessage("总结已完成");
            }
          }
        })
        .catch(() => undefined);
    };
    const timer = window.setInterval(refreshSummary, 2_000);
    refreshSummary();
    return () => window.clearInterval(timer);
  }, [selectedId, summaryRunning]);

  function chooseRecording(recordingId: string) {
    setDetail(null);
    setBlocks([]);
    setCurrentMs(0);
    setFollow(true);
    setShowStartTimeEditor(false);
    setShowSpeakerEditor(false);
    setShowRecordingPicker(false);
    setShowRecordingManager(false);
    setEditingCurrentTitle(false);
    setCurrentTitleMessage("");
    setStartTimeMessage("");
    setSpeakerMessage("");
    setSpeakers([]);
    setFavorites([]);
    setActivity([]);
    setShowSummaryDialog(false);
    setSummaryPromptDraft("");
    setSummaryMessage("");
    setExportMessage("");
    setExportingMarkdown("");
    setSummaryRunning(false);
    setRightView("timeline");
    setSelectedId(recordingId);
  }

  function openRecordingPicker() {
    setRecordingPickerQuery("");
    setActiveRecordingMonth(selected ? recordingMonthKey(selected.recorded_at) : "");
    setShowRecordingPicker(true);
  }

  async function saveInlineRecordingTitle() {
    if (inlineTitleCancelled.current) {
      inlineTitleCancelled.current = false;
      return;
    }
    if (!selected || savingRecording === selected.id) return;
    const title = currentTitleDraft.trim();
    if (!title) {
      setCurrentTitleDraft(selected.title);
      setEditingCurrentTitle(false);
      setCurrentTitleMessage("名称不能为空，已保留原名");
      return;
    }
    if (title === selected.title) {
      setEditingCurrentTitle(false);
      return;
    }
    setSavingRecording(selected.id);
    setEditingCurrentTitle(false);
    try {
      const response = await fetch(`${API}/api/recordings/${selected.id}/title`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      const payload = (await response.json()) as Recording & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法修改录音名称");
      setRecordings((current) => current.map((item) => item.id === payload.id ? { ...item, ...payload } : item));
      setManagedRecordings((current) => current.map((item) => item.id === payload.id ? { ...item, ...payload } : item));
      setRecordingDrafts((current) => ({ ...current, [payload.id]: payload.title }));
      setDetail((current) => current?.id === payload.id ? { ...current, ...payload } : current);
      setCurrentTitleDraft(payload.title);
      setCurrentTitleMessage("");
      await refreshActivityLog(payload.id);
    } catch (caught) {
      setCurrentTitleDraft(selected.title);
      setCurrentTitleMessage(caught instanceof Error ? caught.message : "无法修改录音名称");
    } finally {
      setSavingRecording("");
    }
  }

  const windowStart = Math.floor(currentMs / WINDOW_MS) * WINDOW_MS;
  useEffect(() => {
    if (!selectedId || !selected?.has_transcript) return;
    const controller = new AbortController();
    const end = Math.min(selected.duration_ms, windowStart + WINDOW_MS);
    void fetch(
      `${API}/api/recordings/${selectedId}/blocks?start_ms=${windowStart}&end_ms=${end}`,
      { signal: controller.signal },
    )
      .then((response) => {
        if (!response.ok) throw new Error("无法读取该时段转写");
        return response.json() as Promise<{ blocks: TextBlock[] }>;
      })
      .then((data) => setBlocks(data.blocks))
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(caught.message);
      });
    return () => controller.abort();
  }, [contentRevision, selectedId, selected?.duration_ms, selected?.has_transcript, windowStart]);

  useEffect(() => {
    if (!follow || !activeSentence) return;
    const node = sentenceRefs.current.get(activeSentence.id);
    if (!node) return;
    programmaticScroll.current = true;
    node.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => (programmaticScroll.current = false), 500);
  }, [activeSentence, follow]);

  const seek = useCallback((ms: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, ms / 1000);
    setCurrentMs(Math.max(0, ms));
  }, []);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.code === "Space") {
        event.preventDefault();
        const audio = audioRef.current;
        if (audio) void (audio.paused ? audio.play() : audio.pause());
      }
      if (event.key === "ArrowLeft") seek(currentMs - (event.shiftKey ? 30_000 : 5_000));
      if (event.key === "ArrowRight") seek(currentMs + (event.shiftKey ? 30_000 : 5_000));
    };
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  }, [currentMs, seek]);

  function timelineSeek(event: MouseEvent<HTMLDivElement>) {
    if (!selected) return;
    const rect = event.currentTarget.getBoundingClientRect();
    seek(((event.clientX - rect.left) / rect.width) * selected.duration_ms);
  }

  function openSummaryDialog() {
    if (!selected?.has_transcript || summaryRunning) return;
    const selectedPrompt = summaryPrompts.find((item) => item.id === selectedPromptId);
    setSummaryPromptDraft(selectedPrompt?.prompt ?? detail?.summary?.prompt ?? "");
    setSummaryMessage("");
    setPromptLibraryMessage("");
    setShowPromptEditor(false);
    setShowSummaryDialog(true);
  }

  function chooseSummaryPrompt(prompt: SummaryPrompt) {
    setSelectedPromptId(prompt.id);
    setSummaryPromptDraft(prompt.prompt);
    setPromptLibraryMessage(`已选用“${prompt.name}”，提交时会保存你对内容的修改。`);
  }

  function startNewSummaryPrompt() {
    setPromptNameDraft("");
    setPromptBodyDraft("");
    setPromptLibraryMessage("");
    setShowPromptEditor(true);
  }

  async function saveNewSummaryPrompt() {
    const name = promptNameDraft.trim();
    const prompt = promptBodyDraft.trim();
    if (!name || !prompt) {
      setPromptLibraryMessage("请填写提示词名称和内容后再保存。");
      return;
    }
    setPromptSaving(true);
    setPromptLibraryMessage("");
    try {
      const response = await fetch(`${API}/api/summary-prompts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, prompt }),
      });
      const payload = (await response.json()) as SummaryPrompt & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法保存提示词");
      setSummaryPrompts((current) => [...current, payload]);
      setSelectedPromptId(payload.id);
      setSummaryPromptDraft(payload.prompt);
      setShowPromptEditor(false);
      setPromptLibraryMessage(`“${payload.name}”已保存，可继续修改后提交。`);
    } catch (caught) {
      setPromptLibraryMessage(caught instanceof Error ? caught.message : "无法保存提示词");
    } finally {
      setPromptSaving(false);
    }
  }

  async function saveSelectedSummaryPrompt() {
    if (!selectedPromptId) return;
    const current = summaryPrompts.find((item) => item.id === selectedPromptId);
    if (!current || current.prompt === summaryPromptDraft.trim()) return;
    const response = await fetch(`${API}/api/summary-prompts/${selectedPromptId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: current.name, prompt: summaryPromptDraft }),
    });
    const payload = (await response.json()) as SummaryPrompt & { detail?: string };
    if (!response.ok) throw new Error(payload.detail || "无法保存已修改的提示词");
    setSummaryPrompts((items) => items.map((item) => item.id === payload.id ? payload : item));
  }

  async function submitSummary() {
    if (!selected?.has_transcript || summarySubmitting) return;
    setSummarySubmitting(true);
    setSummaryMessage("");
    try {
      await saveSelectedSummaryPrompt();
      const response = await fetch(`${API}/api/recordings/${selected.id}/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: summaryPromptDraft }),
      });
      const payload = (await response.json()) as { detail?: string; status?: string };
      if (!response.ok) throw new Error(payload.detail || "无法开始总结");
      setShowSummaryDialog(false);
      setSummaryRunning(true);
      setSummaryMessage("已提交给 DeepSeek，正在生成总结…");
      setDetail((current) => current ? {
        ...current,
        summary: {
          status: "queued",
          prompt: summaryPromptDraft,
          progress_percent: 0,
          stage: "等待开始",
        },
      } : current);
    } catch (caught) {
      setSummaryMessage(caught instanceof Error ? caught.message : "无法开始总结");
    } finally {
      setSummarySubmitting(false);
    }
  }

  async function exportMarkdown(kind: "summary" | "transcript") {
    if (!selected?.has_transcript || exportingMarkdown) return;
    if (kind === "summary" && !detail?.topics.length && detail?.summary?.status !== "completed") {
      setExportMessage("总结完成后才能导出总结 Markdown。");
      return;
    }
    setExportingMarkdown(kind);
    setExportMessage("");
    try {
      const response = await fetch(
        `${API}/api/recordings/${selected.id}/export/${kind === "summary" ? "summary" : "transcript"}.md`,
      );
      if (!response.ok) {
        const payload = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(payload.detail || "无法导出 Markdown");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = kind === "summary" ? "总结.md" : "完整对话.md";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setExportMessage(kind === "summary" ? "总结 Markdown 已导出" : "完整对话 Markdown 已导出");
    } catch (caught) {
      setExportMessage(caught instanceof Error ? caught.message : "无法导出 Markdown");
    } finally {
      setExportingMarkdown("");
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMessage(`正在接收 ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    form.append("allow_cloud_upload", String(allowCloud));
    try {
      const response = await fetch(`${API}/api/uploads`, { method: "POST", body: form });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "上传失败");
      setUploadMessage(`已进入转写队列：${payload.id}`);
      setJobs((current) => [payload, ...current.filter((job) => job.id !== payload.id)]);
      void loadJobs();
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? caught.message : "上传失败");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function cancelJob(job: TranscriptionJob) {
    setJobAction(job.id);
    setUploadMessage(`正在取消：${job.filename}。已经产生的云端费用仍可能计费。`);
    setJobs((current) =>
      current.map((item) =>
        item.id === job.id
          ? { ...item, cancel_requested: true, stage: "正在取消" }
          : item,
      ),
    );
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/cancel`, { method: "POST" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法取消转写");
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(`正在取消：${job.filename}`);
    } catch (caught) {
      setJobs((current) =>
        current.map((item) =>
          item.id === job.id
            ? { ...item, cancel_requested: false, stage: job.stage }
            : item,
        ),
      );
      setUploadMessage(caught instanceof Error ? caught.message : "无法取消转写");
    } finally {
      setJobAction("");
    }
  }

  async function dismissJob(job: TranscriptionJob) {
    setJobAction(job.id);
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}`, { method: "DELETE" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法移除任务记录");
      setJobs((current) => current.filter((item) => item.id !== job.id));
      setUploadMessage(`已从任务列表移除：${job.filename}（录音与已有结果仍保留）`);
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? caught.message : "无法移除任务记录");
    } finally {
      setJobAction("");
    }
  }

  async function continueJob(job: TranscriptionJob, decision: RecoveryDecision) {
    setJobAction(job.id);
    setUploadMessage(`正在执行“${decision.continue_label}”：${job.filename}`);
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/continue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: decision.strategy }),
      });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "当前任务无法安全继续");
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(`已按“${decision.continue_label}”进入队列：${job.filename}`);
      void loadJobs();
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? caught.message : "无法继续转写");
    } finally {
      setJobAction("");
    }
  }

  async function cancelFailedJob(job: TranscriptionJob) {
    setJobAction(job.id);
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/decision/cancel`, { method: "POST" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法取消任务");
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(
        job.status === "completed"
          ? `已保留当前转写，暂不修复文本整理：${job.filename}`
          : `已取消并保留已有结果：${job.filename}`,
      );
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? caught.message : "无法取消任务");
    } finally {
      setJobAction("");
    }
  }

  function applyStartTimeUpdate(updated: Detail) {
    setDetail(updated);
    setRecordings((current) =>
      current.map((recording) =>
        recording.id === updated.id ? { ...recording, ...updated } : recording,
      ),
    );
    setStartTimeDraft(dateTimeInputValue(updated.recorded_at));
  }

  function toggleStartTimeEditor() {
    if (!showStartTimeEditor) {
      setStartTimeDraft(dateTimeInputValue(selected?.recorded_at ?? ""));
      setStartTimeMessage("");
    }
    setShowStartTimeEditor((value) => !value);
  }

  async function saveStartTime() {
    if (!selected || !startTimeDraft) return;
    const recordedAt = `${startTimeDraft}${recordingOffset(selected.recorded_at)}`;
    if (Number.isNaN(new Date(recordedAt).getTime())) {
      setStartTimeMessage("请输入有效的开始日期和时间");
      return;
    }
    setSavingStartTime(true);
    setStartTimeMessage("");
    try {
      const response = await fetch(`${API}/api/recordings/${selected.id}/start-time`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recorded_at: recordedAt }),
      });
      const payload = (await response.json()) as Detail & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法保存开始时间");
      applyStartTimeUpdate(payload);
      setStartTimeMessage("已保存，全文实际时间已重新换算");
    } catch (caught) {
      setStartTimeMessage(caught instanceof Error ? caught.message : "无法保存开始时间");
    } finally {
      setSavingStartTime(false);
    }
  }

  async function resetStartTime() {
    if (!selected) return;
    setSavingStartTime(true);
    setStartTimeMessage("");
    try {
      const response = await fetch(`${API}/api/recordings/${selected.id}/start-time`, {
        method: "DELETE",
      });
      const payload = (await response.json()) as Detail & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法恢复文件时间");
      applyStartTimeUpdate(payload);
      setStartTimeMessage("已恢复录音文件记录的开始时间");
    } catch (caught) {
      setStartTimeMessage(caught instanceof Error ? caught.message : "无法恢复文件时间");
    } finally {
      setSavingStartTime(false);
    }
  }

  async function refreshActivityLog(recordingId: string) {
    const response = await fetch(`${API}/api/recordings/${recordingId}/activity?limit=100`);
    if (response.ok) setActivity((await response.json()) as ActivityEvent[]);
  }

  function applySpeakerList(data: SpeakerInfo[]) {
    setSpeakers(data);
    setSpeakerDrafts(
      Object.fromEntries(data.map((speaker) => [speaker.source_speaker, speaker.display_name])),
    );
    setContentRevision((value) => value + 1);
  }

  async function saveSpeaker(sourceSpeaker: string) {
    if (!selected) return;
    const displayName = (speakerDrafts[sourceSpeaker] ?? "").trim();
    if (!displayName) {
      setSpeakerMessage("说话人名称不能为空");
      return;
    }
    setSavingSpeaker(sourceSpeaker);
    setSpeakerMessage("");
    try {
      const response = await fetch(`${API}/api/recordings/${selected.id}/speaker-overrides`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_speaker: sourceSpeaker, display_name: displayName }),
      });
      const payload = (await response.json()) as SpeakerInfo[] & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法保存说话人名称");
      applySpeakerList(payload);
      setSpeakerMessage(`已在整份录音中更新“${sourceSpeaker}”`);
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setSpeakerMessage(caught instanceof Error ? caught.message : "无法保存说话人名称");
    } finally {
      setSavingSpeaker("");
    }
  }

  async function resetSpeaker(sourceSpeaker: string) {
    if (!selected) return;
    setSavingSpeaker(sourceSpeaker);
    setSpeakerMessage("");
    try {
      const query = new URLSearchParams({ source_speaker: sourceSpeaker });
      const response = await fetch(
        `${API}/api/recordings/${selected.id}/speaker-overrides?${query.toString()}`,
        { method: "DELETE" },
      );
      const payload = (await response.json()) as SpeakerInfo[] & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法恢复说话人名称");
      applySpeakerList(payload);
      setSpeakerMessage(`已恢复“${sourceSpeaker}”`);
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setSpeakerMessage(caught instanceof Error ? caught.message : "无法恢复说话人名称");
    } finally {
      setSavingSpeaker("");
    }
  }

  function applyFavorites(data: Favorite[]) {
    setFavorites(data);
    setDetail((current) => (current ? { ...current, favorite_count: data.length } : current));
    setRecordings((current) =>
      current.map((recording) =>
        recording.id === selectedId ? { ...recording, favorite_count: data.length } : recording,
      ),
    );
  }

  async function toggleFavorite(sentence: Sentence) {
    if (!selected) return;
    const isFavorite = favoriteIds.has(sentence.id);
    try {
      const response = await fetch(
        `${API}/api/recordings/${selected.id}/favorites/${sentence.id}`,
        { method: isFavorite ? "DELETE" : "PUT" },
      );
      const payload = (await response.json()) as Favorite[] & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法保存收藏");
      applyFavorites(payload);
      announceFavoriteChange();
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法保存收藏");
    }
  }

  async function refreshRecordingCatalog() {
    await Promise.all([loadRecordings(), loadManagedRecordings(recordingQuery)]);
  }

  async function saveRecordingTitle(recording: Recording) {
    const title = (recordingDrafts[recording.id] ?? "").trim();
    if (!title) {
      setRecordingManagerMessage("录音名称不能为空");
      return;
    }
    setSavingRecording(recording.id);
    setRecordingManagerMessage("");
    try {
      const response = await fetch(`${API}/api/recordings/${recording.id}/title`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      const payload = (await response.json()) as Recording & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法修改录音名称");
      await refreshRecordingCatalog();
      setRecordingManagerMessage(`已将录音显示名改为“${payload.title}”`);
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? caught.message : "无法修改录音名称");
    } finally {
      setSavingRecording("");
    }
  }

  async function resetRecordingTitle(recording: Recording) {
    setSavingRecording(recording.id);
    setRecordingManagerMessage("");
    try {
      const response = await fetch(`${API}/api/recordings/${recording.id}/title`, {
        method: "DELETE",
      });
      const payload = (await response.json()) as Recording & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法恢复录音原名");
      await refreshRecordingCatalog();
      setRecordingManagerMessage(`已恢复原名“${payload.title}”`);
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? caught.message : "无法恢复录音原名");
    } finally {
      setSavingRecording("");
    }
  }

  async function setRecordingHidden(recording: Recording, hidden: boolean) {
    setSavingRecording(recording.id);
    setRecordingManagerMessage("");
    try {
      const response = await fetch(
        hidden
          ? `${API}/api/recordings/${recording.id}`
          : `${API}/api/recordings/${recording.id}/restore`,
        { method: hidden ? "DELETE" : "POST" },
      );
      const payload = (await response.json()) as Recording & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法更新录音目录");
      await refreshRecordingCatalog();
      setRecordingManagerMessage(
        hidden ? `已从工作台移除“${payload.title}”，原文件仍保留` : `已恢复“${payload.title}”`,
      );
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? caught.message : "无法更新录音目录");
    } finally {
      setSavingRecording("");
    }
  }

  if (loading) return <main className="loading-screen">正在整理录音时间线…</main>;

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">声</span>
          <div><h1>录音文本工作台</h1><p>沿着声音，读懂一天</p></div>
        </div>
        <div className="top-actions">
          <button className="recording-picker" onClick={openRecordingPicker} aria-haspopup="dialog">
            <span>当前录音</span>
            <strong>{selected?.title || "选择录音"}</strong>
            <i aria-hidden="true">⌄</i>
          </button>
          <button
            className="quiet-button"
            onClick={() => {
              setShowRecordingManager(true);
              setRecordingQuery("");
              setRecordingManagerMessage("");
              void loadManagedRecordings().catch((caught: Error) => setError(caught.message));
            }}
          >管理录音</button>
          <Link className="quiet-button" href="/favorites" target="_blank" rel="noreferrer">收藏汇总</Link>
          <button className="quiet-button" onClick={() => setShowUpload((value) => !value)}>＋ 添加录音</button>
        </div>
      </header>

      {(showUpload || jobs.length > 0) && (
        <section className="upload-panel">
          {showUpload && (
            <div className="upload-strip">
              <div><strong>添加新的长录音</strong><span>M4A、MP3、WAV 或 FLAC，最大 2 GB</span></div>
              <label className="cloud-choice">
                <input type="checkbox" checked={allowCloud} onChange={(event) => setAllowCloud(event.target.checked)} />
                低成本云端增强（默认开启；仅上传语音区间，费用上限 ¥3；关闭需本地 Qwen3-ASR）
              </label>
              <label className={`upload-button ${uploading ? "disabled" : ""}`}>
                {uploading ? "正在上传…" : "选择录音并开始转写"}
                <input type="file" accept="audio/*,.m4a,.mp3,.wav,.flac" disabled={uploading} onChange={upload} />
              </label>
              {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
            </div>
          )}
          {visibleJobs.length > 0 && (
            <div className="job-board">
              <div className="job-board-heading">
                <div className="job-board-heading-copy">
                  <strong>转写任务</strong>
                  <span>
                    {jobs.length} 条记录
                    {activeJobs.length > 0 && ` · ${activeJobs.length} 条处理中`}
                    {!jobsCollapsed && " · 进度、Token 与费用每 2 秒自动刷新"}
                  </span>
                </div>
                <button
                  type="button"
                  className="job-board-toggle"
                  aria-expanded={!jobsCollapsed}
                  aria-controls="transcription-job-list"
                  onClick={() => setJobsCollapsed((collapsed) => !collapsed)}
                >
                  {jobsCollapsed ? "展开转写任务" : "收起转写任务"}
                </button>
              </div>
              {!jobsCollapsed && <div className="job-list" id="transcription-job-list">
                {visibleJobs.map((job) => (
                  <article className={`job-card ${job.status} text-${job.text_review_status ?? "unknown"}`} key={job.id}>
                    <div className="job-heading">
                      <div><strong>{job.filename}</strong><span>{job.stage}</span></div>
                      <div className="job-heading-side">
                        <b>{job.progress_percent}%</b>
                        <div className="job-actions">
                          {(job.status === "queued" || job.status === "running") ? (
                            <button
                              type="button"
                              className="job-cancel"
                              disabled={jobAction === job.id || job.cancel_requested}
                              onClick={() => void cancelJob(job)}
                              title="取消本地处理；已经产生的云端费用仍可能计费"
                            >
                              {jobAction === job.id || job.cancel_requested ? "正在取消…" : "取消转写"}
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="job-dismiss"
                              disabled={jobAction === job.id}
                              onClick={() => void dismissJob(job)}
                              title="只隐藏任务记录，不删除录音和已有结果"
                            >
                              {jobAction === job.id ? "正在移除…" : "移除记录"}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                    <div
                      className="job-progress"
                      role="progressbar"
                      aria-label={`${job.filename} 处理进度`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={job.progress_percent}
                    >
                      <span style={{ width: `${job.progress_percent}%` }} />
                    </div>
                    <div className="job-cost-grid">
                      <div>
                        <span>处理进度</span>
                        <strong>{job.stage}</strong>
                        <small>
                          {job.retry_count
                            ? `已自动断点重试 ${job.retry_count} 次 · ${job.completed_steps.length} 个阶段已复用`
                            : job.recovery_count
                              ? `服务重启后已恢复 ${job.recovery_count} 次 · ${job.completed_steps.length} 个阶段已复用`
                              : jobCompletionSummary(job)}
                        </small>
                      </div>
                      <div><span>DeepSeek Token</span><strong>{job.text_review_total_tokens ? job.text_review_total_tokens.toLocaleString() : "尚未产生"}</strong><small>输入 {job.text_review_input_tokens.toLocaleString()} / 输出 {job.text_review_output_tokens.toLocaleString()}</small></div>
                      <div><span>Token 费用</span><strong>¥{job.text_review_cost_cny.toFixed(4)}</strong><small>文本整理上限 ¥{job.text_review_cost_cap_cny.toFixed(2)}</small></div>
                      <div><span>ASR 计费</span><strong>¥{job.asr_cost_cny.toFixed(4)}</strong><small>{job.cloud_billed_seconds.toLocaleString()} 秒 · 按音频时长计费</small></div>
                      <div className="job-total-cost"><span>总费用（估算）</span><strong>¥{job.estimated_cost_cny.toFixed(4)}</strong><small>硬上限 ¥{job.cost_cap_cny.toFixed(2)}</small></div>
                    </div>
                    {job.recovery_decision && (
                      <section className="job-decision" aria-label={`${job.filename} 处理决策`}>
                        <div className="job-decision-copy">
                          <span>需要你的决定 · 系统建议</span>
                          <strong>{job.recovery_decision.title}</strong>
                          <p>{job.recovery_decision.description}</p>
                          <small>{job.recovery_decision.impact}</small>
                          {job.recovery_decision.strategy === "extend_text_review_budget" && (
                            <small>选择“追加预算完成剩余窗口”仍只处理文字；也可以保留当前可用结果。</small>
                          )}
                          {jobRecoveryOptions(job).slice(1).map((option) => (
                            <small className="job-alternative" key={option.strategy}>
                              另一种选择：{option.title}。{option.impact}
                            </small>
                          ))}
                        </div>
                        <div className="job-decision-actions">
                          {jobRecoveryOptions(job).map((option, optionIndex) => (
                            <button
                              type="button"
                              className={
                                option.strategy === "continue_cloud_with_higher_cap"
                                  ? "decision-cloud"
                                  : option.strategy === "restart_from_scratch"
                                    ? "decision-restart"
                                  : "decision-continue"
                              }
                              disabled={jobAction === job.id || !option.can_continue}
                              onClick={() => void continueJob(job, option)}
                              title={option.impact}
                              key={option.strategy}
                            >
                              {jobAction === job.id && optionIndex === 0
                                ? "正在处理…"
                                : option.continue_label ||
                                  (job.status === "completed" ? "仅修复文本整理" : "按建议继续")}
                            </button>
                          ))}
                          <button
                            type="button"
                            className="decision-cancel"
                            disabled={jobAction === job.id}
                            onClick={() => void cancelFailedJob(job)}
                          >
                            {job.status === "completed" ? "暂不修复，保留当前结果" : "取消并保留已有结果"}
                          </button>
                        </div>
                      </section>
                    )}
                    {job.retrying && job.last_error && (
                      <p className="job-warning">上次失败：{job.last_error}；正在从已有断点自动续跑。</p>
                    )}
                    {job.status === "failed" && job.error && <p className="job-error">{job.error}</p>}
                    {job.warning && (
                      <p className={job.text_review_status === "partial" ? "job-notice" : "job-warning"}>
                        {job.text_review_status === "fallback" && "完整 ASR 转写已生成；仅文本整理未完成。"}
                        {job.warning}
                        {job.text_review_status === "partial" && "；现有正文和话题可以直接使用。"}
                      </p>
                    )}
                  </article>
                ))}
              </div>}
            </div>
          )}
        </section>
      )}

      {error && <div className="error-banner">{error}。请确认本地服务正在运行。</div>}

      {showRecordingPicker && (
        <div className="speaker-editor-backdrop" onMouseDown={() => setShowRecordingPicker(false)}>
          <section
            className="recording-browser"
            role="dialog"
            aria-modal="true"
            aria-label="选择录音"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading recording-browser-heading">
              <div>
                <span className="eyebrow">RECORDING LIBRARY</span>
                <h2>选择录音</h2>
                <p>按月份浏览；每个月内按当前录音名称排序。</p>
              </div>
              <button aria-label="关闭录音选择" onClick={() => setShowRecordingPicker(false)}>×</button>
            </div>
            <div className="recording-browser-search">
              <label>
                <span>查找录音</span>
                <input
                  type="search"
                  value={recordingPickerQuery}
                  placeholder="输入当前名称或原始文件名"
                  autoFocus
                  onChange={(event) => setRecordingPickerQuery(event.target.value)}
                />
              </label>
              <span>{recordings.length} 份录音</span>
            </div>
            <div className="recording-browser-body">
              <nav className="recording-month-tabs" aria-label="按月份浏览录音">
                {recordingGroups.map((group) => (
                  <button
                    key={group.key}
                    className={`recording-month-tab ${activeRecordingGroup?.key === group.key ? "active" : ""}`}
                    onClick={() => setActiveRecordingMonth(group.key)}
                  >
                    <span>{group.label}</span>
                    <small>{group.recordings.length}</small>
                  </button>
                ))}
              </nav>
              <div className="recording-choice-list">
                {!activeRecordingGroup ? (
                  <div className="saved-empty">没有找到匹配的录音。</div>
                ) : (
                  <>
                    <div className="recording-choice-heading">
                      <strong>{activeRecordingGroup.label}</strong>
                      <span>按名称 A → Z</span>
                    </div>
                    {activeRecordingGroup.recordings.map((recording) => (
                      <button
                        key={recording.id}
                        className={`recording-choice ${recording.id === selectedId ? "active" : ""}`}
                        onClick={() => chooseRecording(recording.id)}
                        title={recordingOptionLabel(recording)}
                      >
                        <span className="recording-choice-mark" aria-hidden="true" />
                        <span className="recording-choice-copy">
                          <strong>{recording.title}</strong>
                          {recording.title_overridden && recording.original_title && (
                            <small>原文件：{recording.original_title}</small>
                          )}
                        </span>
                        <span className="recording-choice-meta">
                          <time>{new Date(recording.recorded_at).toLocaleDateString("zh-CN")}</time>
                          <small>{clock(recording.duration_ms)} · {recording.segment_count.toLocaleString()} 条语音</small>
                        </span>
                      </button>
                    ))}
                  </>
                )}
              </div>
            </div>
            <div className="recording-browser-footer">
              <span>切换录音不会改变名称、时间校准、收藏或转写内容。</span>
              <button onClick={() => setShowRecordingPicker(false)}>关闭</button>
            </div>
          </section>
        </div>
      )}

      {showRecordingManager && (
        <div className="speaker-editor-backdrop" onMouseDown={() => setShowRecordingManager(false)}>
          <section
            className="recording-manager"
            role="dialog"
            aria-modal="true"
            aria-label="管理录音名称"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading">
              <div>
                <span className="eyebrow">RECORDING CATALOG</span>
                <h2>管理录音</h2>
                <p>搜索、重命名、移除或恢复工作台中的录音。</p>
              </div>
              <button aria-label="关闭录音管理" onClick={() => setShowRecordingManager(false)}>×</button>
            </div>
            <div className="recording-manager-note">
              “删除”只从工作台移除，不删除原始音频和转写；隐藏的录音可随时恢复。
            </div>
            <form
              className="recording-search"
              onSubmit={(event) => {
                event.preventDefault();
                void loadManagedRecordings(recordingQuery).catch((caught: Error) =>
                  setRecordingManagerMessage(caught.message),
                );
              }}
            >
              <label>
                <span>搜索录音名称</span>
                <input
                  type="search"
                  value={recordingQuery}
                  placeholder="输入名称或原始文件名"
                  onChange={(event) => setRecordingQuery(event.target.value)}
                />
              </label>
              <button type="submit">查询</button>
              {recordingQuery && (
                <button
                  type="button"
                  className="recording-clear"
                  onClick={() => {
                    setRecordingQuery("");
                    void loadManagedRecordings().catch((caught: Error) =>
                      setRecordingManagerMessage(caught.message),
                    );
                  }}
                >清除</button>
              )}
            </form>
            <div className="recording-manager-list">
              {loadingRecordingCatalog ? (
                <div className="saved-empty">正在读取录音目录…</div>
              ) : !managedRecordings.length ? (
                <div className="saved-empty">没有找到匹配的录音。</div>
              ) : managedRecordings.map((recording) => (
                <div className={`recording-manage-row ${recording.hidden ? "hidden" : ""}`} key={recording.id}>
                  <div className="recording-manage-meta">
                    <span className={recording.hidden ? "recording-state hidden" : "recording-state"}>
                      {recording.hidden ? "已移除" : "使用中"}
                    </span>
                    <small>{clock(recording.duration_ms)} · {recording.segment_count.toLocaleString()} 条语音</small>
                  </div>
                  <label>
                    <span>工作台显示名</span>
                    <input
                      value={recordingDrafts[recording.id] ?? recording.title}
                      maxLength={160}
                      onChange={(event) => {
                        setRecordingDrafts((current) => ({ ...current, [recording.id]: event.target.value }));
                        setRecordingManagerMessage("");
                      }}
                    />
                    {recording.title_overridden && <small>原名：{recording.original_title}</small>}
                  </label>
                  <div className="recording-manage-actions">
                    <button
                      className="recording-save"
                      disabled={savingRecording === recording.id}
                      onClick={() => void saveRecordingTitle(recording)}
                    >保存名称</button>
                    {recording.title_overridden && (
                      <button
                        className="recording-secondary"
                        disabled={savingRecording === recording.id}
                        onClick={() => void resetRecordingTitle(recording)}
                      >恢复原名</button>
                    )}
                    <button
                      className={recording.hidden ? "recording-restore" : "recording-remove"}
                      disabled={savingRecording === recording.id}
                      onClick={() => void setRecordingHidden(recording, !recording.hidden)}
                    >{recording.hidden ? "恢复到工作台" : "从工作台移除"}</button>
                  </div>
                </div>
              ))}
            </div>
            <div className="speaker-editor-footer">
              <span>{recordingManagerMessage || `共显示 ${managedRecordings.length} 份录音`}</span>
              <div className="recording-footer-actions">
                <button
                  className="recording-add"
                  onClick={() => {
                    setShowRecordingManager(false);
                    setShowUpload(true);
                  }}
                >＋ 添加新录音</button>
                <button onClick={() => setShowRecordingManager(false)}>完成</button>
              </div>
            </div>
          </section>
        </div>
      )}

      {showSpeakerEditor && selected && (
        <div className="speaker-editor-backdrop" onMouseDown={() => setShowSpeakerEditor(false)}>
          <section
            className="speaker-editor"
            role="dialog"
            aria-modal="true"
            aria-label="管理整份录音的说话人"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading">
              <div>
                <span className="eyebrow">RECORDING-WIDE SPEAKERS</span>
                <h2>管理说话人</h2>
                <p>{selected.title}</p>
              </div>
              <button aria-label="关闭说话人管理" onClick={() => setShowSpeakerEditor(false)}>×</button>
            </div>
            <div className="speaker-editor-note">
              在这里保存后，会在这份完整录音内统一替换同一原始说话人标签；原始转写不会改变。
            </div>
            <div className="speaker-editor-list">
              {speakers.map((speaker) => (
                <div className="speaker-editor-row" key={speaker.source_speaker}>
                  <div className="speaker-source">
                    <strong>{speaker.source_speaker}</strong>
                    <span>{speaker.segment_count.toLocaleString()} 条语音</span>
                  </div>
                  <label>
                    <span>全局显示为</span>
                    <input
                      value={speakerDrafts[speaker.source_speaker] ?? speaker.display_name}
                      maxLength={80}
                      onChange={(event) => {
                        setSpeakerDrafts((current) => ({
                          ...current,
                          [speaker.source_speaker]: event.target.value,
                        }));
                        setSpeakerMessage("");
                      }}
                    />
                  </label>
                  <div className="speaker-row-actions">
                    <button
                      className="speaker-save"
                      disabled={savingSpeaker === speaker.source_speaker}
                      onClick={() => void saveSpeaker(speaker.source_speaker)}
                    >
                      {savingSpeaker === speaker.source_speaker ? "保存中…" : "保存"}
                    </button>
                    {speaker.is_overridden && (
                      <button
                        className="speaker-reset"
                        disabled={savingSpeaker === speaker.source_speaker}
                        onClick={() => void resetSpeaker(speaker.source_speaker)}
                      >恢复</button>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="speaker-editor-footer">
              <span>{speakerMessage || "每次调整都会写入永久操作日志。"}</span>
              <button onClick={() => setShowSpeakerEditor(false)}>完成</button>
            </div>
          </section>
        </div>
      )}

      {showSummaryDialog && selected && (
        <div className="summary-dialog-backdrop" onMouseDown={() => setShowSummaryDialog(false)}>
          <section
            className="summary-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="summary-dialog-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="summary-dialog-heading">
              <div>
                <span className="eyebrow">DEEPSEEK SUMMARY</span>
                <h2 id="summary-dialog-title">告诉 DS 这次怎么总结</h2>
              </div>
              <button className="modal-close" onClick={() => setShowSummaryDialog(false)} aria-label="关闭总结窗口">×</button>
            </div>
            <p className="summary-dialog-help">
              提示词只影响右侧总结，不会改写原始录音和左侧完整转写。可要求保留全部内容，也可指定只保留某类主题。
            </p>
            {showPromptEditor ? (
              <div className="summary-prompt-editor">
                <label>
                  <span>提示词名称</span>
                  <input
                    value={promptNameDraft}
                    maxLength={80}
                    autoFocus
                    onChange={(event) => setPromptNameDraft(event.target.value)}
                    placeholder="例如：项目复盘"
                  />
                </label>
                <label>
                  <span>提示词内容</span>
                  <textarea
                    className="summary-prompt-input"
                    value={promptBodyDraft}
                    maxLength={4_000}
                    onChange={(event) => setPromptBodyDraft(event.target.value)}
                    placeholder="写清楚要保留、排除和输出的内容。"
                    aria-label="新提示词内容"
                  />
                </label>
                <div className="summary-prompt-editor-actions">
                  <button type="button" className="summary-cancel" onClick={() => setShowPromptEditor(false)}>取消</button>
                  <button type="button" className="summary-submit" disabled={promptSaving} onClick={() => void saveNewSummaryPrompt()}>
                    {promptSaving ? "保存中…" : "保存提示词"}
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="summary-prompt-examples" aria-label="已保存的总结提示词">
                  {summaryPrompts.map((prompt) => (
                    <button
                      key={prompt.id}
                      type="button"
                      className={selectedPromptId === prompt.id ? "active" : ""}
                      onClick={() => chooseSummaryPrompt(prompt)}
                    >
                      {prompt.name}
                    </button>
                  ))}
                  <button type="button" className="summary-prompt-add" onClick={startNewSummaryPrompt}>＋ 添加</button>
                </div>
                <textarea
                  className="summary-prompt-input"
                  value={summaryPromptDraft}
                  maxLength={4_000}
                  autoFocus
                  onChange={(event) => setSummaryPromptDraft(event.target.value)}
                  placeholder="例如：只整理财经观点，去掉感谢礼物、唱歌和闲聊；保留涉及的股票、数据、风险和结论。"
                  aria-label="总结提示词"
                />
                {promptLibraryMessage && <small className="summary-prompt-message">{promptLibraryMessage}</small>}
                <div className="summary-dialog-footer">
                  <span>{summaryPromptDraft.length}/4000 · 已选提示词的修改会在提交时保存</span>
                  <div>
                    <button type="button" className="summary-cancel" onClick={() => setShowSummaryDialog(false)}>取消</button>
                    <button type="button" className="summary-submit" disabled={summarySubmitting} onClick={() => void submitSummary()}>
                      {summarySubmitting ? "提交中…" : "开始总结"}
                    </button>
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      )}

      <div className="content-grid">
        <section className="left-column">
          <div className="player-card">
            <audio
              ref={audioRef}
              src={selectedId ? `${API}/api/recordings/${selectedId}/audio` : undefined}
              preload="metadata"
              onTimeUpdate={(event) => setCurrentMs(event.currentTarget.currentTime * 1000)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
            />
            <div className="player-heading">
              <div className="player-heading-main">
                <span className="recording-status">
                  <span className={`status-dot ${selected?.status === "completed" || selected?.status === "available_with_warning" ? "ready" : "working"}`} />
                  <span>{selected ? statusText(selected) : "未选择录音"}</span>
                </span>
                {selected && (editingCurrentTitle ? (
                  <input
                    className="current-recording-title-input"
                    value={currentTitleDraft}
                    maxLength={160}
                    aria-label="修改当前录音名称"
                    autoFocus
                    onFocus={(event) => event.currentTarget.select()}
                    onChange={(event) => setCurrentTitleDraft(event.target.value)}
                    onBlur={() => void saveInlineRecordingTitle()}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        event.currentTarget.blur();
                      }
                      if (event.key === "Escape") {
                        inlineTitleCancelled.current = true;
                        setCurrentTitleDraft(selected.title);
                        setEditingCurrentTitle(false);
                        event.currentTarget.blur();
                      }
                    }}
                  />
                ) : (
                  <button
                    className="current-recording-title"
                    onClick={() => {
                      inlineTitleCancelled.current = false;
                      setCurrentTitleMessage("");
                      setCurrentTitleDraft(selected.title);
                      setEditingCurrentTitle(true);
                    }}
                    title="单击修改录音名称"
                  >
                    {selected.title}
                    {savingRecording === selected.id && <small>保存中…</small>}
                  </button>
                ))}
                {currentTitleMessage && <span className="current-recording-title-message">{currentTitleMessage}</span>}
              </div>
              <div className="date-tools">
                <span className="date-label">{selected?.recorded_at ? new Date(selected.recorded_at).toLocaleDateString("zh-CN") : ""}</span>
                <button
                  className="speaker-editor-trigger"
                  onClick={() => {
                    setShowSpeakerEditor(true);
                    setSpeakerMessage("");
                  }}
                >管理说话人</button>
                <button
                  className="start-time-trigger"
                  onClick={toggleStartTimeEditor}
                  aria-expanded={showStartTimeEditor}
                >
                  {selected?.start_time_overridden ? "已校准开始时间" : "校准开始时间"}
                </button>
                {showStartTimeEditor && selected && (
                  <div className="start-time-editor">
                    <label>
                      <span>录音当天的实际开始时间</span>
                      <input
                        type="datetime-local"
                        step="1"
                        value={startTimeDraft}
                        onChange={(event) => {
                          setStartTimeDraft(event.target.value);
                          setStartTimeMessage("");
                        }}
                      />
                    </label>
                    <div className="start-time-actions">
                      <button onClick={() => void saveStartTime()} disabled={savingStartTime || !startTimeDraft}>
                        {savingStartTime ? "保存中…" : "保存校准"}
                      </button>
                      {selected.start_time_overridden && (
                        <button className="reset-time-button" onClick={() => void resetStartTime()} disabled={savingStartTime}>
                          恢复文件时间
                        </button>
                      )}
                      <button className="cancel-time-button" onClick={() => setShowStartTimeEditor(false)}>关闭</button>
                    </div>
                    {startTimeMessage && <p>{startTimeMessage}</p>}
                    <small>只校准实际钟表时间，不改变录音内容和录音内进度。</small>
                  </div>
                )}
              </div>
            </div>
            <div className="transport">
              <button aria-label="后退十秒" onClick={() => seek(currentMs - 10_000)}>−10</button>
              <button
                className="play-button"
                aria-label={playing ? "暂停" : "播放"}
                onClick={() => {
                  const audio = audioRef.current;
                  if (audio) void (audio.paused ? audio.play() : audio.pause());
                }}
              >{playing ? "Ⅱ" : "▶"}</button>
              <button aria-label="前进十秒" onClick={() => seek(currentMs + 10_000)}>+10</button>
              <div className="time-readout">
                <strong>{actualClock(selected?.recorded_at ?? "", currentMs)}</strong>
                <span>{clock(currentMs)} / {clock(selected?.duration_ms ?? 0)}</span>
              </div>
              <label className="rate-control">
                <span>速度</span>
                <select value={rate} onChange={(event) => {
                  const value = Number(event.target.value);
                  setRate(value);
                  if (audioRef.current) audioRef.current.playbackRate = value;
                }}>
                  {[0.75, 1, 1.25, 1.5, 2].map((value) => <option key={value} value={value}>{value}×</option>)}
                </select>
              </label>
            </div>
            <div className="timeline-labels"><span>全天概览</span><span>点击任意位置快速定位</span></div>
            <div className="timeline" onClick={timelineSeek} role="slider" aria-label="录音进度" aria-valuenow={currentMs} aria-valuemin={0} aria-valuemax={selected?.duration_ms ?? 0}>
              <div className="density">
                {(detail?.density ?? []).map((value, index) => (
                  <i key={index} style={{ height: `${Math.max(8, value * 100)}%` }} />
                ))}
              </div>
              {detail?.topics.map((topic, index) => (
                <span key={topic.id} className={`topic-band topic-${topic.strength} band-${index % 4}`} style={{ left: `${(topic.start_ms / (selected?.duration_ms || 1)) * 100}%`, width: `${Math.max(0.35, ((topic.end_ms - topic.start_ms) / (selected?.duration_ms || 1)) * 100)}%` }} />
              ))}
              <span className="playhead" style={{ left: `${(currentMs / (selected?.duration_ms || 1)) * 100}%` }} />
            </div>
            <div className="window-ruler">
              <span style={{ width: `${((currentMs % WINDOW_MS) / WINDOW_MS) * 100}%` }} />
            </div>
          </div>

          <div className="transcript-card">
            <div className="section-heading">
              <div><span className="eyebrow">TRANSCRIPT</span><h2>{clock(windowStart)} 至 {clock(Math.min(windowStart + WINDOW_MS, selected?.duration_ms ?? 0))}</h2></div>
              <div className="transcript-heading-actions">
                <button
                  type="button"
                  className="export-button"
                  disabled={!selected?.has_transcript || Boolean(exportingMarkdown)}
                  onClick={() => void exportMarkdown("transcript")}
                >
                  {exportingMarkdown === "transcript" ? "导出中…" : "导出对话 MD"}
                </button>
                <button className={follow ? "follow-button active" : "follow-button"} onClick={() => setFollow(true)}>{follow ? "● 正在跟随" : "回到播放位置"}</button>
              </div>
            </div>
            <div className="uncertainty-note" role="note">
              <span className="uncertainty-swatch" aria-hidden="true" />
              灰色底表示疑似段落，建议回听原音确认。
            </div>
            <div
              className="transcript-scroll"
              ref={transcriptRef}
              onScroll={() => { if (!programmaticScroll.current) setFollow(false); }}
            >
              {!selected?.has_transcript ? (
                <div className="empty-state"><strong>这份录音还在转写</strong><p>完成后，文本会自动出现在这里；现有处理结果和费用记录会继续复用。</p></div>
              ) : blocks.length === 0 ? (
                <div className="empty-state"><strong>这个时段没有可辨识语音</strong><p>可继续拖动上方时间轴查看其他时段。</p></div>
              ) : blocks.map((block) => (
                <article className="text-block" key={block.id}>
                  <button
                    className="block-time"
                    onClick={() => seek(block.start_ms)}
                    title="点击从这段录音开始播放"
                  >
                    <span className="actual-block-time"><span className="time-label">实际</span>{actualRange(selected?.recorded_at ?? "", block.start_ms, block.end_ms)}</span>
                    <span className="elapsed-block-time"><span className="time-label">录音内</span>{clock(block.start_ms)} 至 {clock(block.end_ms)}</span>
                  </button>
                  <p>
                    {block.sentences.map((sentence, index) => {
                      const previous = block.sentences[index - 1];
                      const speakerChanged = !previous || previous.speaker !== sentence.speaker;
                      const isFavorite = favoriteIds.has(sentence.id);
                      return (
                        <span className="sentence-unit" key={sentence.id}>
                          {speakerChanged && <span className="speaker-label">{sentence.speaker}</span>}
                          <button
                            ref={(node) => { if (node) sentenceRefs.current.set(sentence.id, node); else sentenceRefs.current.delete(sentence.id); }}
                            className={`sentence ${activeSentence?.id === sentence.id ? "active" : ""} ${isUncertainSentence(sentence) ? "uncertain" : ""} ${isFavorite ? "favorite" : ""}`}
                            onClick={() => seek(sentence.start_ms)}
                            title={`${clock(sentence.start_ms)} · 点击播放原声`}
                          >{sentence.text}</button>
                          <button
                            className={`favorite-toggle ${isFavorite ? "active" : ""}`}
                            onClick={() => void toggleFavorite(sentence)}
                            aria-label={isFavorite ? "取消收藏这句话" : "收藏这句话"}
                            title={isFavorite ? "取消收藏（日志仍会保留）" : "收藏并高亮这句话"}
                          >{isFavorite ? "★" : "☆"}</button>
                        </span>
                      );
                    })}
                  </p>
                  <details>
                    <summary>查看原始识别与模型候选</summary>
                    <div className="candidate-list">
                      {block.sentences.flatMap((sentence) => sentence.candidates.map((candidate, index) => (
                        <button key={`${sentence.id}-${candidate.provider}-${index}`} onClick={() => seek(sentence.start_ms)}>
                          <span>{clock(sentence.start_ms)} · {candidate.provider === "local" ? "本地" : candidate.provider === "cloud" ? "云端" : "复核"}</span>
                          <p>{candidate.text || "（无文本）"}</p>
                        </button>
                      )))}
                    </div>
                  </details>
                </article>
              ))}
            </div>
          </div>
        </section>

        <aside className="topics-panel">
          <div className="section-heading topics-heading">
            <div>
              <span className="eyebrow">{rightView === "timeline" ? (detail?.topics.length ? (detail?.summary?.source === "text_review" ? "TEXT REVIEW" : "DS SUMMARY") : "SUMMARY") : "SAVED MOMENTS"}</span>
              <h2>{rightView === "timeline" ? (detail?.topics.length ? (detail?.summary?.source === "text_review" ? "初次文本整理" : "总结结果") : "总结") : "收藏与日志"}</h2>
            </div>
            <span>{rightView === "timeline" ? (detail?.topics.length ? `${detail?.summary?.source === "text_review" ? "初次整理 · " : ""}已覆盖 ${detail.topic_segment_count} / ${detail.segment_count} 条语音` : "尚未生成") : `${favorites.length} 条收藏`}</span>
            {rightView === "timeline" && selected?.has_transcript && (detail?.topics.length || detail?.summary?.status === "completed") ? (
              <div className="summary-heading-actions">
                <button
                  type="button"
                  className="export-button"
                  disabled={Boolean(exportingMarkdown) || summaryRunning}
                  onClick={() => void exportMarkdown("summary")}
                >
                  {exportingMarkdown === "summary" ? "导出中…" : "导出总结 MD"}
                </button>
                <button type="button" className="summary-reopen-button" onClick={openSummaryDialog}>重新总结</button>
              </div>
            ) : null}
          </div>
          <div className="right-tabs" role="tablist" aria-label="右侧内容">
            <button className={rightView === "timeline" ? "active" : ""} onClick={() => setRightView("timeline")}>总结</button>
            <button className={rightView === "favorites" ? "active" : ""} onClick={() => setRightView("favorites")}>收藏与日志{favorites.length ? ` ${favorites.length}` : ""}</button>
          </div>
          {rightView === "timeline" ? (
            <>
              {(detail?.summary?.status === "queued" || detail?.summary?.status === "running" || detail?.summary?.status === "failed") && (
                <div className={`summary-status-card ${detail.summary.status === "failed" ? "failed" : ""}`}>
                  <div className="summary-status-card-heading">
                    <strong>{detail.summary.status === "failed" ? "总结没有完成" : detail.summary.stage || "正在准备总结"}</strong>
                    <span>{detail.summary.progress_percent ?? 0}%</span>
                  </div>
                  <div className="summary-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={detail.summary.progress_percent ?? 0}>
                    <span style={{ width: `${detail.summary.progress_percent ?? 0}%` }} />
                  </div>
                  {detail.summary.status === "failed" ? (
                    <>
                      <p>{detail.summary.error || "DeepSeek 暂时没有返回可用结果。"}</p>
                      {detail.summary.can_retry && <button type="button" onClick={openSummaryDialog}>重新尝试</button>}
                    </>
                  ) : (
                    <p>总结在后台进行中，可以继续查看左侧完整转写；完成后这里会自动刷新。</p>
                  )}
                </div>
              )}
              {detail?.topics.length ? (
              <>
                <div className="topic-legend"><span className="strong"><i />按提示词保留</span><span className="weak"><i />未纳入总结的内容仍在左侧转写</span></div>
                <div className="topic-list">
                  {detail.topics.map((topic, index) => (
                  <button key={topic.id} className={`topic-card ${topic.strength} ${activeTopic?.id === topic.id ? "active" : ""}`} onClick={() => seek(topic.start_ms)}>
                    <div className="topic-index">{String(index + 1).padStart(2, "0")}</div>
                    <div className="topic-content">
                      <div className="topic-meta"><span className="topic-time">{clock(topic.start_ms)} 至 {clock(topic.end_ms)}</span><span className="topic-strength">{topic.strength === "strong" ? "明确话题" : "零散 / 弱话题"}</span></div>
                      <h3 title={topic.title}>{topic.title}</h3>
                      <p className="topic-summary" title={topic.summary}>{topic.summary}</p>
                      <div className="keywords">{topic.keywords.slice(0, 5).map((word) => <span key={word} title={word}>{word}</span>)}</div>
                      <small>{topic.segment_count} 条语音，全部按时间顺序包含 · 点击播放</small>
                    </div>
                  </button>
                  ))}
                </div>
              </>
              ) : (
              <div className="summary-empty">
                <span className="summary-empty-icon">✦</span>
                <strong>{detail?.summary?.status === "completed" ? "总结完成，但没有符合要求的内容" : selected?.has_transcript ? "右侧总结尚未生成" : "录音完成转写后可总结"}</strong>
                <p>{detail?.summary?.status === "completed" ? "左侧仍保留完整原始转写，可以换一套提示词重新整理。" : selected?.has_transcript ? "先输入这次录音的整理要求，DeepSeek 会按要求筛选和归纳内容。" : "当前录音还没有可提交给 DeepSeek 的完整转写。"}</p>
                <button type="button" className="summary-open-button" disabled={!selected?.has_transcript || summaryRunning} onClick={openSummaryDialog}>
                  {summaryRunning ? "正在总结…" : detail?.summary?.status === "completed" ? "换提示词重试" : "总结"}
                </button>
                {summaryMessage && <small className="summary-status-message">{summaryMessage}</small>}
              </div>
              )}
            </>
          ) : (
            <div className="saved-panel">
              <section className="favorite-section">
                <div className="saved-section-title"><strong>高亮收藏</strong><span>点击回到原声</span></div>
                {!favorites.length ? (
                  <div className="saved-empty">点击正文旁的 ☆，重要语句会永久保存在这里。</div>
                ) : (
                  <div className="favorite-list">
                    {favorites.map((favorite) => (
                      <button className="favorite-card" key={favorite.segment_id} onClick={() => seek(favorite.start_ms)}>
                        <span><b>★</b>{clock(favorite.start_ms)} · {favorite.speaker}</span>
                        <p>{favorite.text}</p>
                      </button>
                    ))}
                  </div>
                )}
              </section>
              <section className="activity-section">
                <div className="saved-section-title"><strong>永久操作日志</strong><span>仅追加，不随取消收藏删除</span></div>
                {!activity.length ? (
                  <div className="saved-empty">还没有说话人调整或收藏操作。</div>
                ) : (
                  <div className="activity-list">
                    {activity.map((event) => (
                      <div className="activity-row" key={event.event_id}>
                        <i />
                        <div><strong>{activityText(event)}</strong><span>{new Date(event.created_at).toLocaleString("zh-CN")}</span></div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
          {exportMessage && <div className="export-message" role="status">{exportMessage}</div>}
          <div className="shortcut-note"><strong>快捷键</strong><span>空格播放 / 暂停</span><span>← → 跳转 5 秒</span><span>Shift + ← → 跳转 30 秒</span></div>
        </aside>
      </div>
    </main>
  );
}
