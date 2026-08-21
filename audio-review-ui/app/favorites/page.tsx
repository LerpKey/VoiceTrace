"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { announceFavoriteChange, FAVORITES_CHANNEL } from "../favorite-events";

const API = process.env.NEXT_PUBLIC_AUDIO_API ?? "http://127.0.0.1:8765";

type FavoriteSummary = {
  recording_id: string;
  recording_title: string;
  recorded_at: string;
  segment_id: string;
  start_ms: number;
  end_ms: number;
  speaker: string;
  text: string;
  created_at: string;
  note: string;
  note_updated_at: string | null;
};

function elapsed(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, "0")).join(":");
}

function actualTime(recordedAt: string, offsetMs: number) {
  const date = new Date(new Date(recordedAt).getTime() + offsetMs);
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function recordingDate(recordedAt: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(new Date(recordedAt));
}

export default function FavoritesPage() {
  const [favorites, setFavorites] = useState<FavoriteSummary[]>([]);
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState("");
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const loadFavorites = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch(`${API}/api/favorites`, { signal });
      if (!response.ok) throw new Error("无法读取收藏汇总");
      setFavorites((await response.json()) as FavoriteSummary[]);
      setError("");
    } catch (caught) {
      if (caught instanceof Error && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "无法读取收藏汇总");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${API}/api/favorites`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("无法读取收藏汇总");
        return response.json() as Promise<FavoriteSummary[]>;
      })
      .then((data) => {
        setFavorites(data);
        setError("");
      })
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(caught.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    const channel =
      typeof BroadcastChannel === "undefined"
        ? null
        : new BroadcastChannel(FAVORITES_CHANNEL);
    if (channel) channel.onmessage = () => void loadFavorites();
    const refreshWhenActive = () => {
      if (document.visibilityState === "visible") void loadFavorites();
    };
    window.addEventListener("focus", refreshWhenActive);
    document.addEventListener("visibilitychange", refreshWhenActive);
    const refreshTimer = window.setInterval(() => void loadFavorites(), 5_000);
    return () => {
      controller.abort();
      channel?.close();
      window.removeEventListener("focus", refreshWhenActive);
      document.removeEventListener("visibilitychange", refreshWhenActive);
      window.clearInterval(refreshTimer);
    };
  }, [loadFavorites]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    if (!normalized) return favorites;
    return favorites.filter((favorite) =>
      [favorite.recording_title, favorite.speaker, favorite.text, favorite.note]
        .join(" ")
        .toLocaleLowerCase("zh-CN")
        .includes(normalized),
    );
  }, [favorites, query]);

  const groups = useMemo(() => {
    const grouped = new Map<string, FavoriteSummary[]>();
    for (const favorite of filtered) {
      const items = grouped.get(favorite.recording_id) ?? [];
      items.push(favorite);
      grouped.set(favorite.recording_id, items);
    }
    return [...grouped.values()];
  }, [filtered]);

  const recordingCount = new Set(favorites.map((favorite) => favorite.recording_id)).size;
  const noteCount = favorites.filter((favorite) => favorite.note).length;

  async function saveNote(favorite: FavoriteSummary) {
    const key = `${favorite.recording_id}:${favorite.segment_id}`;
    setSaving(key);
    setError("");
    try {
      const response = await fetch(
        `${API}/api/recordings/${favorite.recording_id}/favorites/${favorite.segment_id}/note`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ note: drafts[key] ?? "" }),
        },
      );
      const payload = (await response.json()) as FavoriteSummary & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || "无法保存备注");
      setFavorites((current) =>
        current.map((item) =>
          item.recording_id === favorite.recording_id && item.segment_id === favorite.segment_id
            ? { ...item, note: payload.note, note_updated_at: payload.note_updated_at }
            : item,
        ),
      );
      setDrafts((current) => ({ ...current, [key]: payload.note }));
      setEditing("");
      announceFavoriteChange();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法保存备注");
    } finally {
      setSaving("");
    }
  }

  function beginEditing(favorite: FavoriteSummary) {
    const key = `${favorite.recording_id}:${favorite.segment_id}`;
    setDrafts((current) => ({ ...current, [key]: favorite.note }));
    setEditing(key);
  }

  async function refreshFavorites() {
    setRefreshing(true);
    await loadFavorites();
    setRefreshing(false);
  }

  return (
    <main className="favorites-workspace">
      <header className="favorites-header">
        <div className="brand">
          <span className="brand-mark">藏</span>
          <div><h1>收藏语句汇总</h1><p>跨越所有录音，集中整理值得保留的片段</p></div>
        </div>
        <div className="favorites-header-actions">
          <button
            className="quiet-button"
            type="button"
            disabled={refreshing}
            onClick={() => void refreshFavorites()}
          >{refreshing ? "刷新中…" : "刷新收藏"}</button>
          <Link className="quiet-button" href="/">返回录音工作台</Link>
        </div>
      </header>

      <section className="favorites-overview" aria-label="收藏统计">
        <div><strong>{favorites.length}</strong><span>收藏语句</span></div>
        <div><strong>{recordingCount}</strong><span>份录音</span></div>
        <div><strong>{noteCount}</strong><span>条备注</span></div>
        <label className="favorites-search">
          <span>搜索收藏、说话人或备注</span>
          <input
            type="search"
            value={query}
            placeholder="输入关键词"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      </section>

      {error && <div className="favorites-error">{error}</div>}

      <div className="favorites-content">
        {loading ? (
          <div className="favorites-empty"><strong>正在汇总收藏…</strong></div>
        ) : groups.length === 0 ? (
          <div className="favorites-empty">
            <strong>{query ? "没有匹配的收藏" : "还没有收藏语句"}</strong>
            <p>{query ? "换一个关键词试试。" : "回到录音工作台，点击句子旁的星标即可收藏。"}</p>
          </div>
        ) : groups.map((items) => {
          const recording = items[0];
          return (
            <section className="favorite-recording-group" key={recording.recording_id}>
              <div className="favorite-group-heading">
                <div>
                  <span>{recordingDate(recording.recorded_at)}</span>
                  <h2>{recording.recording_title}</h2>
                </div>
                <strong>{items.length} 条</strong>
              </div>
              <div className="favorite-summary-list">
                {items.map((favorite) => {
                  const key = `${favorite.recording_id}:${favorite.segment_id}`;
                  const isEditing = editing === key;
                  return (
                    <article className="favorite-summary-card" key={key}>
                      <div className="favorite-sentence-meta">
                        <span>{favorite.speaker}</span>
                        <time>
                          实际 {actualTime(favorite.recorded_at, favorite.start_ms)} · 录音内 {elapsed(favorite.start_ms)}
                        </time>
                      </div>
                      <p className="favorite-sentence-text">{favorite.text}</p>
                      <div className="favorite-note-area">
                        <span className="favorite-note-label">备注</span>
                        {isEditing ? (
                          <div className="favorite-note-editor">
                            <textarea
                              autoFocus
                              maxLength={2_000}
                              value={drafts[key] ?? ""}
                              placeholder="写下后续事项、背景或你的想法"
                              onChange={(event) =>
                                setDrafts((current) => ({
                                  ...current,
                                  [key]: event.target.value,
                                }))
                              }
                            />
                            <div>
                              <small>{(drafts[key] ?? "").length} / 2000</small>
                              <button
                                className="note-cancel"
                                type="button"
                                onClick={() => setEditing("")}
                              >取消</button>
                              <button
                                type="button"
                                disabled={saving === key}
                                onClick={() => void saveNote(favorite)}
                              >{saving === key ? "保存中…" : "保存备注"}</button>
                            </div>
                          </div>
                        ) : (
                          <button
                            className={`favorite-note-preview ${favorite.note ? "has-note" : ""}`}
                            type="button"
                            title={favorite.note || "点击添加备注"}
                            onClick={() => beginEditing(favorite)}
                          >
                            {favorite.note || "＋ 点击添加备注"}
                          </button>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </main>
  );
}

