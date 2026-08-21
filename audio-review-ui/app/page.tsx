"use client";

import Link from "next/link";
import { ChangeEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { announceFavoriteChange } from "./favorite-events";
import {
  formatMessage,
  LanguageSwitcher,
  localizeServerText,
  localizedPromptName,
  localizedStage,
  Locale,
  useLocale,
} from "./locale";

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
    name: "Complete meeting",
    prompt: "Review the meeting in recording order. Keep the source language of the recording and transcript; do not translate it. Include topics, decisions, owners, numbers, and action items. Preserve uncertainty and do not invent missing details.",
  },
  {
    id: "prompt-finance-live",
    name: "Finance livestream",
    prompt: "Keep only the host's finance-related content: market views, stocks and sectors, macro data, risks, and reasoning. Remove gifts, greetings, singing, small talk, and other unrelated content. Keep the source language of the recording and transcript; do not translate it.",
  },
];

const UNCERTAIN_FLAGS = new Set(["models_disagree", "no_majority", "sensitive_difference", "vad_supplement"]);

function isUncertainSentence(sentence: Sentence) {
  return sentence.text.startsWith("[Uncertain:") || sentence.text.startsWith("[Unclear]") || sentence.flags.some((flag) => UNCERTAIN_FLAGS.has(flag));
}

function recordingOptionLabel(recording: Recording, locale: Locale) {
  if (
    recording.title_overridden &&
    recording.original_title &&
    recording.original_title !== recording.title
  ) {
    return `${recording.title} (${formatMessage(locale, "originalFile", { name: recording.original_title })})`;
  }
  return recording.title;
}

function recordingMonthKey(recordedAt: string) {
  const date = new Date(recordedAt);
  if (Number.isNaN(date.getTime())) return "unknown";
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

function recordingMonthLabel(monthKey: string, locale: Locale) {
  if (monthKey === "unknown") return formatMessage(locale, "dateUnknown");
  const [year, month] = monthKey.split("-");
  return formatMessage(locale, "monthLabel", { year, month: Number(month) });
}

function sortRecordingsByTitle(left: Recording, right: Recording, locale: Locale) {
  return left.title.localeCompare(right.title, locale === "zh-CN" ? "zh-CN" : "en", { numeric: true, sensitivity: "base" });
}

function jobCompletionSummary(job: TranscriptionJob, locale: Locale) {
  if (job.core_transcript_ready && job.text_review_status === "fallback") {
    return formatMessage(locale, "coreReadyPending", { count: job.completed_steps.length });
  }
  if (job.core_transcript_ready && job.text_review_status === "partial") {
    return formatMessage(locale, "coreReadyPartial", { count: job.completed_steps.length });
  }
  return formatMessage(locale, "stagesComplete", { count: job.completed_steps.length });
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

function actualClock(recordedAt: string, ms: number, locale: Locale) {
  const date = new Date(new Date(recordedAt).getTime() + ms);
  return Number.isNaN(date.getTime())
    ? clock(ms)
    : new Intl.DateTimeFormat(locale === "zh-CN" ? "zh-CN" : "en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(date);
}

function actualRange(recordedAt: string, startMs: number, endMs: number, locale: Locale) {
  const recordedTime = new Date(recordedAt).getTime();
  if (Number.isNaN(recordedTime)) return formatMessage(locale, "actualTimeUnknown");

  const start = new Date(recordedTime + startMs);
  const end = new Date(recordedTime + endMs);
  const crossesDay =
    start.getFullYear() !== end.getFullYear() ||
    start.getMonth() !== end.getMonth() ||
    start.getDate() !== end.getDate();
  const formatter = new Intl.DateTimeFormat(locale === "zh-CN" ? "zh-CN" : "en-US", {
    ...(crossesDay ? { month: "2-digit", day: "2-digit" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  return `${formatter.format(start)} – ${formatter.format(end)}`;
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

function statusText(recording: Recording, locale: Locale) {
  const count = recording.segment_count.toLocaleString(locale === "zh-CN" ? "zh-CN" : "en-US");
  if (recording.status === "completed") return formatMessage(locale, "statusCompleted", { count });
  if (recording.status === "available_with_warning") return formatMessage(locale, "statusWarning", { count });
  if (recording.status === "failed") return formatMessage(locale, "statusFailed");
  return formatMessage(locale, "statusProcessing");
}

function activityText(event: ActivityEvent, locale: Locale) {
  const details = event.details;
  if (event.action === "favorite_added") return formatMessage(locale, "activityFavoriteAdded", { time: clock(Number(details.start_ms ?? 0)) });
  if (event.action === "favorite_removed") return formatMessage(locale, "activityFavoriteRemoved", { time: clock(Number(details.start_ms ?? 0)) });
  if (event.action === "favorite_note_updated") return formatMessage(locale, "activityNoteUpdated");
  if (event.action === "favorite_note_cleared") return formatMessage(locale, "activityNoteCleared");
  if (event.action === "speaker_override_removed") return formatMessage(locale, "activitySpeakerReset", { speaker: String(details.source_speaker ?? "Speaker") });
  if (event.action === "speaker_override_updated") {
    return formatMessage(locale, "activitySpeakerUpdated", { speaker: String(details.source_speaker ?? "Speaker"), name: String(details.display_name ?? "") });
  }
  if (event.action === "recording_title_updated") return formatMessage(locale, "activityTitleUpdated", { title: String(details.title ?? "") });
  if (event.action === "recording_title_reset") return formatMessage(locale, "activityTitleReset", { title: String(details.title ?? "") });
  if (event.action === "recording_hidden") return formatMessage(locale, "activityHidden", { title: String(details.title ?? "") });
  if (event.action === "recording_restored") return formatMessage(locale, "activityRestored", { title: String(details.title ?? "") });
  return event.action;
}

export default function Home() {
  const { locale, setLocale, t } = useLocale();
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
    const query = recordingPickerQuery.trim().toLocaleLowerCase(locale === "zh-CN" ? "zh-CN" : "en-US");
    const matches = recordings.filter((recording) =>
      !query || recording.title.toLocaleLowerCase(locale === "zh-CN" ? "zh-CN" : "en-US").includes(query)
        || recording.original_title?.toLocaleLowerCase(locale === "zh-CN" ? "zh-CN" : "en-US").includes(query),
    );
    const grouped = new Map<string, Recording[]>();
    [...matches].sort((left, right) => sortRecordingsByTitle(left, right, locale)).forEach((recording) => {
      const month = recordingMonthKey(recording.recorded_at);
      grouped.set(month, [...(grouped.get(month) ?? []), recording]);
    });
    return [...grouped.entries()]
      .sort(([left], [right]) => {
        if (left === "unknown") return 1;
        if (right === "unknown") return -1;
        return right.localeCompare(left);
      })
      .map(([key, items]) => ({ key, label: recordingMonthLabel(key, locale), recordings: items }));
  }, [locale, recordingPickerQuery, recordings]);
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
      throw new Error(t("readSpeakersFavorites"));
    }
    return {
      speakers: (await speakerResponse.json()) as SpeakerInfo[],
      favorites: (await favoriteResponse.json()) as Favorite[],
      activity: (await activityResponse.json()) as ActivityEvent[],
    };
  }, [t]);

  const loadRecordings = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/recordings`);
      if (!response.ok) throw new Error(t("serviceUnavailable"));
      const data = (await response.json()) as Recording[];
      setRecordings(data);
      setSelectedId((current) =>
        data.some((recording) => recording.id === current)
          ? current
          : data.find((item) => item.has_transcript)?.id || data[0]?.id || "",
      );
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? localizeServerText(locale, caught.message) : t("readRecordings"));
    } finally {
      setLoading(false);
    }
  }, [locale, t]);

  const loadJobs = useCallback(async () => {
    const response = await fetch(`${API}/api/jobs`);
    if (!response.ok) throw new Error(t("readJobs"));
    const data = (await response.json()) as TranscriptionJob[];
    setJobs(data);
    if (data.some((job) => job.status === "queued" || job.status === "running")) {
      setShowUpload(true);
    }
  }, [t]);

  const loadSummaryPrompts = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/summary-prompts`);
      if (!response.ok) throw new Error(t("readPrompts"));
      const data = (await response.json()) as SummaryPrompt[];
      if (data.length) setSummaryPrompts(data);
    } catch {
      // Keep the two local starter prompts available when the API is temporarily offline.
    }
  }, [t]);

  const loadManagedRecordings = useCallback(async (query = "") => {
    setLoadingRecordingCatalog(true);
    setRecordingManagerMessage("");
    try {
      const params = new URLSearchParams({ include_hidden: "true" });
      if (query.trim()) params.set("q", query.trim());
      const response = await fetch(`${API}/api/recordings?${params.toString()}`);
      if (!response.ok) throw new Error(t("readCatalog"));
      const data = (await response.json()) as Recording[];
      setManagedRecordings(data);
      setRecordingDrafts(
        Object.fromEntries(data.map((recording) => [recording.id, recording.title])),
      );
    } finally {
      setLoadingRecordingCatalog(false);
    }
  }, [t]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadSummaryPrompts(), 0);
    return () => window.clearTimeout(timer);
  }, [loadSummaryPrompts]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${API}/api/recordings`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(t("serviceUnavailable"));
        return response.json() as Promise<Recording[]>;
      })
      .then((data) => {
        setRecordings(data);
        setSelectedId(data.find((item) => item.has_transcript)?.id || data[0]?.id || "");
        setError("");
      })
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(localizeServerText(locale, caught.message));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [locale, t]);

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
        if (!response.ok) throw new Error(t("readDetail"));
        return response.json() as Promise<Detail>;
      })
      .then((data) => {
        setDetail(data);
        setSummaryRunning(data.summary?.status === "queued" || data.summary?.status === "running");
      })
      .catch((caught: Error) => setError(localizeServerText(locale, caught.message)));
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
      .catch((caught: Error) => setError(localizeServerText(locale, caught.message)));
  }, [loadRecordingExtras, locale, selectedId, t]);

  useEffect(() => {
    if (!summaryRunning || !selectedId) return;
    const refreshSummary = () => {
      void fetch(`${API}/api/recordings/${selectedId}`)
        .then((response) => {
          if (!response.ok) throw new Error(t("readSummaryProgress"));
          return response.json() as Promise<Detail>;
        })
        .then((data) => {
          setDetail(data);
          if (data.summary?.status !== "queued" && data.summary?.status !== "running") {
            setSummaryRunning(false);
            if (data.summary?.status === "failed") {
              setSummaryMessage(localizeServerText(locale, data.summary.error) || t("summaryFailedCheck"));
            } else {
              setSummaryMessage(t("summaryCompleted"));
            }
          }
        })
        .catch(() => undefined);
    };
    const timer = window.setInterval(refreshSummary, 2_000);
    refreshSummary();
    return () => window.clearInterval(timer);
  }, [locale, selectedId, summaryRunning, t]);

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
      setCurrentTitleMessage(t("nameRequired"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveRecordingNameError"));
      setRecordings((current) => current.map((item) => item.id === payload.id ? { ...item, ...payload } : item));
      setManagedRecordings((current) => current.map((item) => item.id === payload.id ? { ...item, ...payload } : item));
      setRecordingDrafts((current) => ({ ...current, [payload.id]: payload.title }));
      setDetail((current) => current?.id === payload.id ? { ...current, ...payload } : current);
      setCurrentTitleDraft(payload.title);
      setCurrentTitleMessage("");
      await refreshActivityLog(payload.id);
    } catch (caught) {
      setCurrentTitleDraft(selected.title);
      setCurrentTitleMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveRecordingNameError"));
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
        if (!response.ok) throw new Error(t("readBlocks"));
        return response.json() as Promise<{ blocks: TextBlock[] }>;
      })
      .then((data) => setBlocks(data.blocks))
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(localizeServerText(locale, caught.message));
      });
    return () => controller.abort();
  }, [contentRevision, locale, selectedId, selected?.duration_ms, selected?.has_transcript, t, windowStart]);

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
    setPromptLibraryMessage(formatMessage(locale, "promptSelected", { name: localizedPromptName(locale, prompt.id, prompt.name) }));
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
      setPromptLibraryMessage(t("promptRequired"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("savePromptError"));
      setSummaryPrompts((current) => [...current, payload]);
      setSelectedPromptId(payload.id);
      setSummaryPromptDraft(payload.prompt);
      setShowPromptEditor(false);
      setPromptLibraryMessage(formatMessage(locale, "promptSaved", { name: payload.name }));
    } catch (caught) {
      setPromptLibraryMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("savePromptError"));
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
    if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("savePromptError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("startSummaryError"));
      setShowSummaryDialog(false);
      setSummaryRunning(true);
      setSummaryMessage(t("summarySubmitted"));
      setDetail((current) => current ? {
        ...current,
        summary: {
          status: "queued",
          prompt: summaryPromptDraft,
          progress_percent: 0,
          stage: t("summaryWaiting"),
        },
      } : current);
    } catch (caught) {
      setSummaryMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("startSummaryError"));
    } finally {
      setSummarySubmitting(false);
    }
  }

  async function exportMarkdown(kind: "summary" | "transcript") {
    if (!selected?.has_transcript || exportingMarkdown) return;
    if (kind === "summary" && !detail?.topics.length && detail?.summary?.status !== "completed") {
      setExportMessage(t("summaryNotReady"));
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
        throw new Error(localizeServerText(locale, payload.detail) || t("exportError"));
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = kind === "summary" ? "summary.md" : "transcript.md";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setExportMessage(kind === "summary" ? t("summaryExported") : t("transcriptExported"));
    } catch (caught) {
      setExportMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("exportError"));
    } finally {
      setExportingMarkdown("");
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMessage(formatMessage(locale, "receivingFile", { name: file.name }));
    const form = new FormData();
    form.append("file", file);
    form.append("allow_cloud_upload", String(allowCloud));
    try {
      const response = await fetch(`${API}/api/uploads`, { method: "POST", body: form });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("uploadFailed"));
      setUploadMessage(formatMessage(locale, "queuedFile", { id: payload.id }));
      setJobs((current) => [payload, ...current.filter((job) => job.id !== payload.id)]);
      void loadJobs();
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("uploadFailed"));
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function cancelJob(job: TranscriptionJob) {
    setJobAction(job.id);
    setUploadMessage(formatMessage(locale, "cancellingFile", { name: job.filename }));
    setJobs((current) =>
      current.map((item) =>
        item.id === job.id
          ? { ...item, cancel_requested: true, stage: "Cancelling" }
          : item,
      ),
    );
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/cancel`, { method: "POST" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("cancelTranscriptionError"));
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(formatMessage(locale, "cancellingFileShort", { name: job.filename }));
    } catch (caught) {
      setJobs((current) =>
        current.map((item) =>
          item.id === job.id
            ? { ...item, cancel_requested: false, stage: job.stage }
            : item,
        ),
      );
      setUploadMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("cancelTranscriptionError"));
    } finally {
      setJobAction("");
    }
  }

  async function dismissJob(job: TranscriptionJob) {
    setJobAction(job.id);
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}`, { method: "DELETE" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("removeTaskError"));
      setJobs((current) => current.filter((item) => item.id !== job.id));
      setUploadMessage(formatMessage(locale, "taskRemoved", { name: job.filename }));
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("removeTaskError"));
    } finally {
      setJobAction("");
    }
  }

  async function continueJob(job: TranscriptionJob, decision: RecoveryDecision) {
    setJobAction(job.id);
    const localizedDecision = localizedRecoveryDecision(locale, decision);
    setUploadMessage(formatMessage(locale, "executingDecision", { label: localizedDecision.continue_label, name: job.filename }));
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/continue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: decision.strategy }),
      });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("continueTaskError"));
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(formatMessage(locale, "queuedDecision", { label: localizedDecision.continue_label, name: job.filename }));
      void loadJobs();
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("continueTaskError"));
    } finally {
      setJobAction("");
    }
  }

  async function cancelFailedJob(job: TranscriptionJob) {
    setJobAction(job.id);
    try {
      const response = await fetch(`${API}/api/jobs/${job.id}/decision/cancel`, { method: "POST" });
      const payload = (await response.json()) as TranscriptionJob & { detail?: string };
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("cancelTaskError"));
      setJobs((current) => current.map((item) => (item.id === job.id ? payload : item)));
      setUploadMessage(
        job.status === "completed"
          ? formatMessage(locale, "keepTranscription", { name: job.filename })
          : formatMessage(locale, "cancelKeepResults", { name: job.filename }),
      );
    } catch (caught) {
      setUploadMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("cancelTaskError"));
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
      setStartTimeMessage(t("invalidStartTime"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveStartTimeError"));
      applyStartTimeUpdate(payload);
      setStartTimeMessage(t("startTimeSaved"));
    } catch (caught) {
      setStartTimeMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveStartTimeError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("resetStartTimeError"));
      applyStartTimeUpdate(payload);
      setStartTimeMessage(t("startTimeRestored"));
    } catch (caught) {
      setStartTimeMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("resetStartTimeError"));
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
      setSpeakerMessage(t("speakerNameRequired"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveSpeakerError"));
      applySpeakerList(payload);
      setSpeakerMessage(formatMessage(locale, "speakerUpdated", { speaker: sourceSpeaker }));
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setSpeakerMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveSpeakerError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("resetSpeakerError"));
      applySpeakerList(payload);
      setSpeakerMessage(formatMessage(locale, "speakerRestored", { speaker: sourceSpeaker }));
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setSpeakerMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("resetSpeakerError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveFavoriteError"));
      applyFavorites(payload);
      announceFavoriteChange();
      await refreshActivityLog(selected.id);
    } catch (caught) {
      setError(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveFavoriteError"));
    }
  }

  async function refreshRecordingCatalog() {
    await Promise.all([loadRecordings(), loadManagedRecordings(recordingQuery)]);
  }

  async function saveRecordingTitle(recording: Recording) {
    const title = (recordingDrafts[recording.id] ?? "").trim();
    if (!title) {
      setRecordingManagerMessage(t("nameRequired"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveRecordingNameError"));
      await refreshRecordingCatalog();
      setRecordingManagerMessage(formatMessage(locale, "recordingRenamed", { title: payload.title }));
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveRecordingNameError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("resetRecordingNameError"));
      await refreshRecordingCatalog();
      setRecordingManagerMessage(formatMessage(locale, "recordingNameRestored", { title: payload.title }));
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("resetRecordingNameError"));
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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("updateRecordingCatalogError"));
      await refreshRecordingCatalog();
      setRecordingManagerMessage(
        hidden ? formatMessage(locale, "recordingHidden", { title: payload.title }) : formatMessage(locale, "recordingRestored", { title: payload.title }),
      );
    } catch (caught) {
      setRecordingManagerMessage(caught instanceof Error ? localizeServerText(locale, caught.message) : t("updateRecordingCatalogError"));
    } finally {
      setSavingRecording("");
    }
  }

  if (loading) return <main className="loading-screen">{t("loadingTimeline")}</main>;

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">V</span>
          <div><h1>VoiceTrace</h1><p>{t("brandTagline")}</p></div>
        </div>
        <div className="top-actions">
          <button className="recording-picker" onClick={openRecordingPicker} aria-haspopup="dialog">
            <span>{t("currentRecording")}</span>
            <strong>{selected?.title || t("selectRecording")}</strong>
            <i aria-hidden="true">⌄</i>
          </button>
          <LanguageSwitcher locale={locale} onChange={setLocale} />
          <button
            className="quiet-button"
            onClick={() => {
              setShowRecordingManager(true);
              setRecordingQuery("");
              setRecordingManagerMessage("");
              void loadManagedRecordings().catch((caught: Error) => setError(caught.message));
            }}
          >{t("manageRecordings")}</button>
          <Link className="quiet-button" href="/favorites" target="_blank" rel="noreferrer">{t("favoritesSummary")}</Link>
          <button className="quiet-button" onClick={() => setShowUpload((value) => !value)}>{t("addRecording")}</button>
        </div>
      </header>

      {(showUpload || jobs.length > 0) && (
        <section className="upload-panel">
          {showUpload && (
            <div className="upload-strip">
              <div><strong>{t("addLongRecording")}</strong><span>{t("supportedAudio")}</span></div>
              <label className="cloud-choice">
                <input type="checkbox" checked={allowCloud} onChange={(event) => setAllowCloud(event.target.checked)} />
                {t("cloudChoice")}
              </label>
              <label className={`upload-button ${uploading ? "disabled" : ""}`}>
                {uploading ? t("uploading") : t("chooseAndStart")}
                <input type="file" accept="audio/*,.m4a,.mp3,.wav,.flac" disabled={uploading} onChange={upload} />
              </label>
              {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
            </div>
          )}
          {visibleJobs.length > 0 && (
            <div className="job-board">
              <div className="job-board-heading">
                <div className="job-board-heading-copy">
                  <strong>{t("transcriptionJobs")}</strong>
                  <span>
                    {formatMessage(locale, "records", { count: jobs.length })}
                    {activeJobs.length > 0 && ` · ${formatMessage(locale, "activeProcessing", { count: activeJobs.length })}`}
                    {!jobsCollapsed && ` · ${t("refreshDetails")}`}
                  </span>
                </div>
                <button
                  type="button"
                  className="job-board-toggle"
                  aria-expanded={!jobsCollapsed}
                  aria-controls="transcription-job-list"
                  onClick={() => setJobsCollapsed((collapsed) => !collapsed)}
                >
                  {jobsCollapsed ? t("expandJobs") : t("collapseJobs")}
                </button>
              </div>
              {!jobsCollapsed && <div className="job-list" id="transcription-job-list">
                {visibleJobs.map((job) => {
                  const recoveryOptions = jobRecoveryOptions(job).map((option) => localizedRecoveryDecision(locale, option));
                  const recoveryDecision = job.recovery_decision
                    ? localizedRecoveryDecision(locale, job.recovery_decision)
                    : null;
                  return (
                  <article className={`job-card ${job.status} text-${job.text_review_status ?? "unknown"}`} key={job.id}>
                    <div className="job-heading">
                      <div><strong>{job.filename}</strong><span>{localizedStage(locale, job.stage)}</span></div>
                      <div className="job-heading-side">
                        <b>{job.progress_percent}%</b>
                        <div className="job-actions">
                          {(job.status === "queued" || job.status === "running") ? (
                            <button
                              type="button"
                              className="job-cancel"
                              disabled={jobAction === job.id || job.cancel_requested}
                              onClick={() => void cancelJob(job)}
                              title={t("cancelLocalTitle")}
                            >
                              {jobAction === job.id || job.cancel_requested ? t("cancelling") : t("cancelTranscription")}
                            </button>
                          ) : (
                            <button
                              type="button"
                              className="job-dismiss"
                              disabled={jobAction === job.id}
                              onClick={() => void dismissJob(job)}
                              title={t("hideTaskTitle")}
                            >
                              {jobAction === job.id ? t("removing") : t("removeRecord")}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                    <div
                      className="job-progress"
                      role="progressbar"
                      aria-label={`${job.filename} ${t("processingProgress")}`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={job.progress_percent}
                    >
                      <span style={{ width: `${job.progress_percent}%` }} />
                    </div>
                    <div className="job-cost-grid">
                      <div>
                        <span>{t("processingProgress")}</span>
                        <strong>{localizedStage(locale, job.stage)}</strong>
                        <small>
                          {job.retry_count
                            ? formatMessage(locale, "checkpointRetried", { count: job.retry_count, stages: job.completed_steps.length })
                            : job.recovery_count
                              ? formatMessage(locale, "restartRecovered", { count: job.recovery_count, stages: job.completed_steps.length })
                              : jobCompletionSummary(job, locale)}
                        </small>
                      </div>
                      <div><span>DeepSeek Token</span><strong>{job.text_review_total_tokens ? job.text_review_total_tokens.toLocaleString(locale) : t("tokensNotGenerated")}</strong><small>{formatMessage(locale, "tokenBreakdown", { input: job.text_review_input_tokens.toLocaleString(locale), output: job.text_review_output_tokens.toLocaleString(locale) })}</small></div>
                      <div><span>{t("tokenCost")}</span><strong>¥{job.text_review_cost_cny.toFixed(4)}</strong><small>{formatMessage(locale, "textReviewCap", { cap: job.text_review_cost_cap_cny.toFixed(2) })}</small></div>
                      <div><span>{t("asrBilling")}</span><strong>¥{job.asr_cost_cny.toFixed(4)}</strong><small>{formatMessage(locale, "billedSeconds", { seconds: job.cloud_billed_seconds.toLocaleString(locale) })}</small></div>
                      <div className="job-total-cost"><span>{t("estimatedTotal")}</span><strong>¥{job.estimated_cost_cny.toFixed(4)}</strong><small>{formatMessage(locale, "hardCap", { cap: job.cost_cap_cny.toFixed(2) })}</small></div>
                    </div>
                    {recoveryDecision && (
                      <section className="job-decision" aria-label={`${job.filename} ${t("decisionRequired")}`}>
                        <div className="job-decision-copy">
                          <span>{t("decisionRequired")}</span>
                          <strong>{recoveryDecision.title}</strong>
                          <p>{recoveryDecision.description}</p>
                          <small>{recoveryDecision.impact}</small>
                          {recoveryDecision.strategy === "extend_text_review_budget" && (
                            <small>{t("extendTextReviewNote")}</small>
                          )}
                          {recoveryOptions.slice(1).map((option) => (
                            <small className="job-alternative" key={option.strategy}>
                              {formatMessage(locale, "alternative", { title: option.title, impact: option.impact })}
                            </small>
                          ))}
                        </div>
                        <div className="job-decision-actions">
                          {recoveryOptions.map((option, optionIndex) => (
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
                                ? t("processing")
                                : option.continue_label ||
                                  (job.status === "completed" ? t("repairTextOnly") : t("continueAsRecommended"))}
                            </button>
                          ))}
                          <button
                            type="button"
                            className="decision-cancel"
                            disabled={jobAction === job.id}
                            onClick={() => void cancelFailedJob(job)}
                          >
                            {job.status === "completed" ? t("keepCurrentResult") : t("cancelKeepExisting")}
                          </button>
                        </div>
                      </section>
                    )}
                    {job.retrying && job.last_error && (
                      <p className="job-warning">{formatMessage(locale, "previousFailure", { error: localizeServerText(locale, job.last_error) })}</p>
                    )}
                    {job.status === "failed" && job.error && <p className="job-error">{localizeServerText(locale, job.error)}</p>}
                    {job.warning && (
                      <p className={job.text_review_status === "partial" ? "job-notice" : "job-warning"}>
                        {job.text_review_status === "fallback" && t("fullTranscriptReady")}
                        {localizeServerText(locale, job.warning)}
                        {job.text_review_status === "partial" && `; ${t("usableTranscript")}`}
                      </p>
                    )}
                  </article>
                  );
                })}
              </div>}
            </div>
          )}
        </section>
      )}

      {error && <div className="error-banner">{error}. {t("noLocalService")}</div>}

      {showRecordingPicker && (
        <div className="speaker-editor-backdrop" onMouseDown={() => setShowRecordingPicker(false)}>
          <section
            className="recording-browser"
            role="dialog"
            aria-modal="true"
            aria-label={t("chooseRecordingTitle")}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading recording-browser-heading">
              <div>
                <span className="eyebrow">{t("recordingLibrary")}</span>
                <h2>{t("chooseRecordingTitle")}</h2>
                <p>{t("browseMonths")}</p>
              </div>
              <button aria-label={t("closeRecordingPicker")} onClick={() => setShowRecordingPicker(false)}>×</button>
            </div>
            <div className="recording-browser-search">
              <label>
                <span>{t("findRecording")}</span>
                <input
                  type="search"
                  value={recordingPickerQuery}
                  placeholder={t("nameOrOriginal")}
                  autoFocus
                  onChange={(event) => setRecordingPickerQuery(event.target.value)}
                />
              </label>
              <span>{formatMessage(locale, "recordingsCount", { count: recordings.length })}</span>
            </div>
            <div className="recording-browser-body">
              <nav className="recording-month-tabs" aria-label={t("browseMonthsAria")}>
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
                  <div className="saved-empty">{t("noRecordingMatches")}</div>
                ) : (
                  <>
                    <div className="recording-choice-heading">
                      <strong>{activeRecordingGroup.label}</strong>
                      <span>{t("sortedAZ")}</span>
                    </div>
                    {activeRecordingGroup.recordings.map((recording) => (
                      <button
                        key={recording.id}
                        className={`recording-choice ${recording.id === selectedId ? "active" : ""}`}
                        onClick={() => chooseRecording(recording.id)}
                        title={recordingOptionLabel(recording, locale)}
                      >
                        <span className="recording-choice-mark" aria-hidden="true" />
                        <span className="recording-choice-copy">
                          <strong>{recording.title}</strong>
                          {recording.title_overridden && recording.original_title && (
                            <small>{formatMessage(locale, "originalFile", { name: recording.original_title })}</small>
                          )}
                        </span>
                        <span className="recording-choice-meta">
                          <time>{new Date(recording.recorded_at).toLocaleDateString(locale)}</time>
                          <small>{formatMessage(locale, "recordingMeta", { duration: clock(recording.duration_ms), count: recording.segment_count.toLocaleString(locale) })}</small>
                        </span>
                      </button>
                    ))}
                  </>
                )}
              </div>
            </div>
            <div className="recording-browser-footer">
              <span>{t("switchingRecordingNote")}</span>
              <button onClick={() => setShowRecordingPicker(false)}>{t("close")}</button>
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
            aria-label={t("manageRecordingTitle")}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading">
              <div>
                <span className="eyebrow">{t("recordingCatalog")}</span>
                <h2>{t("manageRecordingTitle")}</h2>
                <p>{t("manageRecordingDescription")}</p>
              </div>
              <button aria-label={t("closeRecordingManager")} onClick={() => setShowRecordingManager(false)}>×</button>
            </div>
            <div className="recording-manager-note">
              {t("removeDoesNotDelete")}
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
                <span>{t("searchRecordingName")}</span>
                <input
                  type="search"
                  value={recordingQuery}
                  placeholder={t("nameOrOriginalShort")}
                  onChange={(event) => setRecordingQuery(event.target.value)}
                />
              </label>
              <button type="submit">{t("query")}</button>
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
                >{t("clear")}</button>
              )}
            </form>
            <div className="recording-manager-list">
              {loadingRecordingCatalog ? (
                <div className="saved-empty">{t("readingCatalog")}</div>
              ) : !managedRecordings.length ? (
                <div className="saved-empty">{t("noRecordingMatches")}</div>
              ) : managedRecordings.map((recording) => (
                <div className={`recording-manage-row ${recording.hidden ? "hidden" : ""}`} key={recording.id}>
                  <div className="recording-manage-meta">
                    <span className={recording.hidden ? "recording-state hidden" : "recording-state"}>
                      {recording.hidden ? t("hidden") : t("active")}
                    </span>
                    <small>{formatMessage(locale, "recordingMeta", { duration: clock(recording.duration_ms), count: recording.segment_count.toLocaleString(locale) })}</small>
                  </div>
                  <label>
                    <span>{t("workspaceDisplayName")}</span>
                    <input
                      value={recordingDrafts[recording.id] ?? recording.title}
                      maxLength={160}
                      onChange={(event) => {
                        setRecordingDrafts((current) => ({ ...current, [recording.id]: event.target.value }));
                        setRecordingManagerMessage("");
                      }}
                    />
                    {recording.title_overridden && <small>{formatMessage(locale, "originalName", { name: recording.original_title ?? "" })}</small>}
                  </label>
                  <div className="recording-manage-actions">
                    <button
                      className="recording-save"
                      disabled={savingRecording === recording.id}
                      onClick={() => void saveRecordingTitle(recording)}
                    >{savingRecording === recording.id ? t("saving") : t("saveName")}</button>
                    {recording.title_overridden && (
                      <button
                        className="recording-secondary"
                        disabled={savingRecording === recording.id}
                        onClick={() => void resetRecordingTitle(recording)}
                      >{t("restoreOriginal")}</button>
                    )}
                    <button
                      className={recording.hidden ? "recording-restore" : "recording-remove"}
                      disabled={savingRecording === recording.id}
                      onClick={() => void setRecordingHidden(recording, !recording.hidden)}
                    >{recording.hidden ? t("restoreWorkspace") : t("removeFromWorkspace")}</button>
                  </div>
                </div>
              ))}
            </div>
            <div className="speaker-editor-footer">
              <span>{recordingManagerMessage || formatMessage(locale, "displayedRecordings", { count: managedRecordings.length })}</span>
              <div className="recording-footer-actions">
                <button
                  className="recording-add"
                  onClick={() => {
                    setShowRecordingManager(false);
                    setShowUpload(true);
                  }}
                >{t("addNewRecording")}</button>
                <button onClick={() => setShowRecordingManager(false)}>{t("done")}</button>
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
            aria-label={t("manageSpeakers")}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="speaker-editor-heading">
              <div>
                <span className="eyebrow">{t("recordingWideSpeakers")}</span>
                <h2>{t("manageSpeakers")}</h2>
                <p>{selected.title}</p>
              </div>
              <button aria-label={t("closeSpeakerManager")} onClick={() => setShowSpeakerEditor(false)}>×</button>
            </div>
            <div className="speaker-editor-note">
              {t("speakerInstruction")}
            </div>
            <div className="speaker-editor-list">
              {speakers.map((speaker) => (
                <div className="speaker-editor-row" key={speaker.source_speaker}>
                  <div className="speaker-source">
                    <strong>{speaker.source_speaker}</strong>
                    <span>{formatMessage(locale, "speechSegments", { count: speaker.segment_count.toLocaleString(locale) })}</span>
                  </div>
                  <label>
                    <span>{t("displayAs")}</span>
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
                      {savingSpeaker === speaker.source_speaker ? t("saving") : t("saveName")}
                    </button>
                    {speaker.is_overridden && (
                      <button
                        className="speaker-reset"
                        disabled={savingSpeaker === speaker.source_speaker}
                        onClick={() => void resetSpeaker(speaker.source_speaker)}
                      >{t("restore")}</button>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="speaker-editor-footer">
              <span>{speakerMessage || t("speakerLogNote")}</span>
              <button onClick={() => setShowSpeakerEditor(false)}>{t("done")}</button>
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
                <h2 id="summary-dialog-title">{t("summaryHow")}</h2>
              </div>
              <button className="modal-close" onClick={() => setShowSummaryDialog(false)} aria-label={t("closeSummary")}>×</button>
            </div>
            <p className="summary-dialog-help">
              {t("summaryHelp")}
            </p>
            {showPromptEditor ? (
              <div className="summary-prompt-editor">
                <label>
                  <span>{t("promptName")}</span>
                  <input
                    value={promptNameDraft}
                    maxLength={80}
                    autoFocus
                    onChange={(event) => setPromptNameDraft(event.target.value)}
                    placeholder={t("newPromptExample")}
                  />
                </label>
                <label>
                  <span>{t("promptContent")}</span>
                  <textarea
                    className="summary-prompt-input"
                    value={promptBodyDraft}
                    maxLength={4_000}
                    onChange={(event) => setPromptBodyDraft(event.target.value)}
                    placeholder={t("promptContentPlaceholder")}
                    aria-label={t("promptContent")}
                  />
                </label>
                <div className="summary-prompt-editor-actions">
                  <button type="button" className="summary-cancel" onClick={() => setShowPromptEditor(false)}>{t("summaryCancel")}</button>
                  <button type="button" className="summary-submit" disabled={promptSaving} onClick={() => void saveNewSummaryPrompt()}>
                    {promptSaving ? t("saving") : t("savePrompt")}
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="summary-prompt-examples" aria-label={t("savedSummaryPrompts")}>
                  {summaryPrompts.map((prompt) => (
                    <button
                      key={prompt.id}
                      type="button"
                      className={selectedPromptId === prompt.id ? "active" : ""}
                      onClick={() => chooseSummaryPrompt(prompt)}
                    >
                      {localizedPromptName(locale, prompt.id, prompt.name)}
                    </button>
                  ))}
                  <button type="button" className="summary-prompt-add" onClick={startNewSummaryPrompt}>{t("addPrompt")}</button>
                </div>
                <textarea
                  className="summary-prompt-input"
                  value={summaryPromptDraft}
                  maxLength={4_000}
                  autoFocus
                  onChange={(event) => setSummaryPromptDraft(event.target.value)}
                  placeholder={t("summaryPromptPlaceholder")}
                  aria-label={t("summaryPromptAria")}
                />
                {promptLibraryMessage && <small className="summary-prompt-message">{promptLibraryMessage}</small>}
                <div className="summary-dialog-footer">
                  <span>{formatMessage(locale, "promptCount", { count: summaryPromptDraft.length })}</span>
                  <div>
                    <button type="button" className="summary-cancel" onClick={() => setShowSummaryDialog(false)}>{t("summaryCancel")}</button>
                    <button type="button" className="summary-submit" disabled={summarySubmitting} onClick={() => void submitSummary()}>
                      {summarySubmitting ? t("submitting") : t("startSummary")}
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
                  <span>{selected ? statusText(selected, locale) : t("statusNone")}</span>
                </span>
                {selected && (editingCurrentTitle ? (
                  <input
                    className="current-recording-title-input"
                    value={currentTitleDraft}
                    maxLength={160}
                    aria-label={t("editCurrentTitle")}
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
                    title={t("editCurrentTitle")}
                  >
                    {selected.title}
                    {savingRecording === selected.id && <small>{t("saving")}</small>}
                  </button>
                ))}
                {currentTitleMessage && <span className="current-recording-title-message">{currentTitleMessage}</span>}
              </div>
              <div className="date-tools">
                <span className="date-label">{selected?.recorded_at ? new Date(selected.recorded_at).toLocaleDateString(locale) : ""}</span>
                <button
                  className="speaker-editor-trigger"
                  onClick={() => {
                    setShowSpeakerEditor(true);
                    setSpeakerMessage("");
                  }}
                >{t("manageSpeakers")}</button>
                <button
                  className="start-time-trigger"
                  onClick={toggleStartTimeEditor}
                  aria-expanded={showStartTimeEditor}
                >
                  {selected?.start_time_overridden ? t("calibratedStart") : t("calibrateStart")}
                </button>
                {showStartTimeEditor && selected && (
                  <div className="start-time-editor">
                    <label>
                      <span>{t("actualStartTime")}</span>
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
                        {savingStartTime ? t("saving") : t("saveCalibration")}
                      </button>
                      {selected.start_time_overridden && (
                        <button className="reset-time-button" onClick={() => void resetStartTime()} disabled={savingStartTime}>
                          {t("restoreFileTime")}
                        </button>
                      )}
                      <button className="cancel-time-button" onClick={() => setShowStartTimeEditor(false)}>{t("close")}</button>
                    </div>
                    {startTimeMessage && <p>{startTimeMessage}</p>}
                    <small>{t("calibrationNote")}</small>
                  </div>
                )}
              </div>
            </div>
            <div className="transport">
              <button aria-label={t("backwardTen")} onClick={() => seek(currentMs - 10_000)}>−10</button>
              <button
                className="play-button"
                aria-label={playing ? t("pause") : t("play")}
                onClick={() => {
                  const audio = audioRef.current;
                  if (audio) void (audio.paused ? audio.play() : audio.pause());
                }}
              >{playing ? "Ⅱ" : "▶"}</button>
              <button aria-label={t("forwardTen")} onClick={() => seek(currentMs + 10_000)}>+10</button>
              <div className="time-readout">
                <strong>{actualClock(selected?.recorded_at ?? "", currentMs, locale)}</strong>
                <span>{clock(currentMs)} / {clock(selected?.duration_ms ?? 0)}</span>
              </div>
              <label className="rate-control">
                <span>{t("speed")}</span>
                <select value={rate} onChange={(event) => {
                  const value = Number(event.target.value);
                  setRate(value);
                  if (audioRef.current) audioRef.current.playbackRate = value;
                }}>
                  {[0.75, 1, 1.25, 1.5, 2].map((value) => <option key={value} value={value}>{value}×</option>)}
                </select>
              </label>
            </div>
            <div className="timeline-labels"><span>{t("allDayOverview")}</span><span>{t("timelineHint")}</span></div>
            <div className="timeline" onClick={timelineSeek} role="slider" aria-label={t("recordingProgress")} aria-valuenow={currentMs} aria-valuemin={0} aria-valuemax={selected?.duration_ms ?? 0}>
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
              <div><span className="eyebrow">{t("transcript")}</span><h2>{clock(windowStart)} – {clock(Math.min(windowStart + WINDOW_MS, selected?.duration_ms ?? 0))}</h2></div>
              <div className="transcript-heading-actions">
                <button
                  type="button"
                  className="export-button"
                  disabled={!selected?.has_transcript || Boolean(exportingMarkdown)}
                  onClick={() => void exportMarkdown("transcript")}
                >
                  {exportingMarkdown === "transcript" ? t("exporting") : t("exportTranscript")}
                </button>
                <button className={follow ? "follow-button active" : "follow-button"} onClick={() => setFollow(true)}>{follow ? t("following") : t("returnToPlayback")}</button>
              </div>
            </div>
            <div className="uncertainty-note" role="note">
              <span className="uncertainty-swatch" aria-hidden="true" />
              {t("uncertaintyNote")}
            </div>
            <div
              className="transcript-scroll"
              ref={transcriptRef}
              onScroll={() => { if (!programmaticScroll.current) setFollow(false); }}
            >
              {!selected?.has_transcript ? (
                <div className="empty-state"><strong>{t("transcriptPending")}</strong><p>{t("transcriptPendingDescription")}</p></div>
              ) : blocks.length === 0 ? (
                <div className="empty-state"><strong>{t("noSpeech")}</strong><p>{t("noSpeechDescription")}</p></div>
              ) : blocks.map((block) => (
                <article className="text-block" key={block.id}>
                  <button
                    className="block-time"
                    onClick={() => seek(block.start_ms)}
                    title={t("playFromSegment")}
                  >
                    <span className="actual-block-time"><span className="time-label">{t("actual")}</span>{actualRange(selected?.recorded_at ?? "", block.start_ms, block.end_ms, locale)}</span>
                    <span className="elapsed-block-time"><span className="time-label">{t("elapsed")}</span>{clock(block.start_ms)} – {clock(block.end_ms)}</span>
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
                            title={`${clock(sentence.start_ms)} · ${t("play")}`}
                          >{sentence.text}</button>
                          <button
                            className={`favorite-toggle ${isFavorite ? "active" : ""}`}
                            onClick={() => void toggleFavorite(sentence)}
                            aria-label={isFavorite ? t("removeFavoriteSentence") : t("favoriteSentence")}
                            title={isFavorite ? t("removeFavoriteTitle") : t("favoriteSentenceTitle")}
                          >{isFavorite ? "★" : "☆"}</button>
                        </span>
                      );
                    })}
                  </p>
                  <details>
                    <summary>{t("originalCandidates")}</summary>
                    <div className="candidate-list">
                      {block.sentences.flatMap((sentence) => sentence.candidates.map((candidate, index) => (
                        <button key={`${sentence.id}-${candidate.provider}-${index}`} onClick={() => seek(sentence.start_ms)}>
                          <span>{clock(sentence.start_ms)} · {candidate.provider === "local" ? t("local") : candidate.provider === "cloud" ? t("cloud") : t("review")}</span>
                          <p>{candidate.text || t("noText")}</p>
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
              <span className="eyebrow">{rightView === "timeline" ? (detail?.topics.length ? (detail?.summary?.source === "text_review" ? t("initialTextReview") : t("summaryResult")) : t("summary")) : t("savedMoments")}</span>
              <h2>{rightView === "timeline" ? (detail?.topics.length ? (detail?.summary?.source === "text_review" ? t("initialTextReview") : t("summaryResult")) : t("summary")) : t("savedAndLogs")}</h2>
            </div>
            <span>{rightView === "timeline" ? (detail?.topics.length ? formatMessage(locale, "coverage", { prefix: detail?.summary?.source === "text_review" ? `${t("initialTextReview")} · ` : "", covered: detail.topic_segment_count, total: detail.segment_count }) : t("notGenerated")) : formatMessage(locale, "favoriteCount", { count: favorites.length })}</span>
            {rightView === "timeline" && selected?.has_transcript && (detail?.topics.length || detail?.summary?.status === "completed") ? (
              <div className="summary-heading-actions">
                <button
                  type="button"
                  className="export-button"
                  disabled={Boolean(exportingMarkdown) || summaryRunning}
                  onClick={() => void exportMarkdown("summary")}
                >
                  {exportingMarkdown === "summary" ? t("exporting") : t("exportSummary")}
                </button>
                <button type="button" className="summary-reopen-button" onClick={openSummaryDialog}>{t("resummarize")}</button>
              </div>
            ) : null}
          </div>
          <div className="right-tabs" role="tablist" aria-label={t("rightContent")}>
            <button className={rightView === "timeline" ? "active" : ""} onClick={() => setRightView("timeline")}>{t("summaryTab")}</button>
            <button className={rightView === "favorites" ? "active" : ""} onClick={() => setRightView("favorites")}>{formatMessage(locale, "savedTab", { count: favorites.length ? ` ${favorites.length}` : "" })}</button>
          </div>
          {rightView === "timeline" ? (
            <>
              {(detail?.summary?.status === "queued" || detail?.summary?.status === "running" || detail?.summary?.status === "failed") && (
                <div className={`summary-status-card ${detail.summary.status === "failed" ? "failed" : ""}`}>
                  <div className="summary-status-card-heading">
                    <strong>{detail.summary.status === "failed" ? t("summaryFailed") : localizeServerText(locale, detail.summary.stage) || t("preparingSummary")}</strong>
                    <span>{detail.summary.progress_percent ?? 0}%</span>
                  </div>
                  <div className="summary-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={detail.summary.progress_percent ?? 0}>
                    <span style={{ width: `${detail.summary.progress_percent ?? 0}%` }} />
                  </div>
                  {detail.summary.status === "failed" ? (
                    <>
                      <p>{localizeServerText(locale, detail.summary.error) || t("deepseekNoResult")}</p>
                      {detail.summary.can_retry && <button type="button" onClick={openSummaryDialog}>{t("retrySummary")}</button>}
                    </>
                  ) : (
                    <p>{t("summaryInProgress")}</p>
                  )}
                </div>
              )}
              {detail?.topics.length ? (
              <>
                <div className="topic-legend"><span className="strong"><i />{t("keepByPrompt")}</span><span className="weak"><i />{t("omittedOnLeft")}</span></div>
                <div className="topic-list">
                  {detail.topics.map((topic, index) => (
                  <button key={topic.id} className={`topic-card ${topic.strength} ${activeTopic?.id === topic.id ? "active" : ""}`} onClick={() => seek(topic.start_ms)}>
                    <div className="topic-index">{String(index + 1).padStart(2, "0")}</div>
                    <div className="topic-content">
                      <div className="topic-meta"><span className="topic-time">{clock(topic.start_ms)} – {clock(topic.end_ms)}</span><span className="topic-strength">{topic.strength === "strong" ? t("strongTopic") : t("weakTopic")}</span></div>
                      <h3 title={topic.title}>{topic.title}</h3>
                      <p className="topic-summary" title={topic.summary}>{topic.summary}</p>
                      <div className="keywords">{topic.keywords.slice(0, 5).map((word) => <span key={word} title={word}>{word}</span>)}</div>
                      <small>{formatMessage(locale, "topicMeta", { count: topic.segment_count })}</small>
                    </div>
                  </button>
                  ))}
                </div>
              </>
              ) : (
              <div className="summary-empty">
                <span className="summary-empty-icon">✦</span>
                <strong>{detail?.summary?.status === "completed" ? t("completedNoContent") : selected?.has_transcript ? t("rightSummaryNotGenerated") : t("canSummarizeAfterTranscript")}</strong>
                <p>{detail?.summary?.status === "completed" ? t("completedNoContentDescription") : selected?.has_transcript ? t("summaryNotGeneratedDescription") : t("canSummarizeDescription")}</p>
                <button type="button" className="summary-open-button" disabled={!selected?.has_transcript || summaryRunning} onClick={openSummaryDialog}>
                  {summaryRunning ? t("summarizing") : detail?.summary?.status === "completed" ? t("changePromptRetry") : t("summary")}
                </button>
                {summaryMessage && <small className="summary-status-message">{summaryMessage}</small>}
              </div>
              )}
            </>
          ) : (
            <div className="saved-panel">
              <section className="favorite-section">
                <div className="saved-section-title"><strong>{t("highlightedFavorites")}</strong><span>{t("clickToOriginal")}</span></div>
                {!favorites.length ? (
                  <div className="saved-empty">{t("noFavorites")}</div>
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
                <div className="saved-section-title"><strong>{t("permanentLog")}</strong><span>{t("logAppendOnly")}</span></div>
                {!activity.length ? (
                  <div className="saved-empty">{t("noActivity")}</div>
                ) : (
                  <div className="activity-list">
                    {activity.map((event) => (
                      <div className="activity-row" key={event.event_id}>
                        <i />
                        <div><strong>{activityText(event, locale)}</strong><span>{new Date(event.created_at).toLocaleString(locale)}</span></div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
          {exportMessage && <div className="export-message" role="status">{exportMessage}</div>}
          <div className="shortcut-note"><strong>{t("shortcuts")}</strong><span>{t("shortcutPlay")}</span><span>{t("shortcutFive")}</span><span>{t("shortcutThirty")}</span></div>
        </aside>
      </div>
    </main>
  );
}
