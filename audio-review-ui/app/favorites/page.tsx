"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { announceFavoriteChange, FAVORITES_CHANNEL } from "../favorite-events";
import { LanguageSwitcher, localizeServerText, Locale, useLocale } from "../locale";

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

function actualTime(recordedAt: string, offsetMs: number, locale: Locale) {
  const date = new Date(new Date(recordedAt).getTime() + offsetMs);
  return new Intl.DateTimeFormat(locale === "zh-CN" ? "zh-CN" : "en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function recordingDate(recordedAt: string, locale: Locale) {
  return new Intl.DateTimeFormat(locale === "zh-CN" ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(new Date(recordedAt));
}

export default function FavoritesPage() {
  const { locale, setLocale, t } = useLocale();
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
      if (!response.ok) throw new Error(t("readFavoriteSummary"));
      setFavorites((await response.json()) as FavoriteSummary[]);
      setError("");
    } catch (caught) {
      if (caught instanceof Error && caught.name === "AbortError") return;
      setError(caught instanceof Error ? localizeServerText(locale, caught.message) : t("readFavoriteSummary"));
    }
  }, [locale, t]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${API}/api/favorites`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(t("readFavoriteSummary"));
        return response.json() as Promise<FavoriteSummary[]>;
      })
      .then((data) => {
        setFavorites(data);
        setError("");
      })
      .catch((caught: Error) => {
        if (caught.name !== "AbortError") setError(localizeServerText(locale, caught.message));
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
  }, [loadFavorites, locale, t]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase(locale === "zh-CN" ? "zh-CN" : "en-US");
    if (!normalized) return favorites;
    return favorites.filter((favorite) =>
      [favorite.recording_title, favorite.speaker, favorite.text, favorite.note]
        .join(" ")
        .toLocaleLowerCase(locale === "zh-CN" ? "zh-CN" : "en-US")
        .includes(normalized),
    );
  }, [favorites, locale, query]);

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
      if (!response.ok) throw new Error(localizeServerText(locale, payload.detail) || t("saveNoteError"));
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
      setError(caught instanceof Error ? localizeServerText(locale, caught.message) : t("saveNoteError"));
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
          <span className="brand-mark">V</span>
          <div><h1>{t("favoritesTitle")}</h1><p>{t("favoritesSubtitle")}</p></div>
        </div>
        <div className="favorites-header-actions">
          <LanguageSwitcher locale={locale} onChange={setLocale} />
          <button
            className="quiet-button"
            type="button"
            disabled={refreshing}
            onClick={() => void refreshFavorites()}
          >{refreshing ? t("refreshing") : t("refreshFavorites")}</button>
          <Link className="quiet-button" href="/">{t("backToWorkspace")}</Link>
        </div>
      </header>

      <section className="favorites-overview" aria-label={t("favoriteStats")}>
        <div><strong>{favorites.length}</strong><span>{t("favoriteSentences")}</span></div>
        <div><strong>{recordingCount}</strong><span>{t("recordingCount")}</span></div>
        <div><strong>{noteCount}</strong><span>{t("noteCount")}</span></div>
        <label className="favorites-search">
          <span>{t("searchFavorites")}</span>
          <input
            type="search"
            value={query}
            placeholder={t("keywordPlaceholder")}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      </section>

      {error && <div className="favorites-error">{error}</div>}

      <div className="favorites-content">
        {loading ? (
          <div className="favorites-empty"><strong>{t("aggregatingFavorites")}</strong></div>
        ) : groups.length === 0 ? (
          <div className="favorites-empty">
            <strong>{query ? t("noMatchingFavorites") : t("noFavoriteSentences")}</strong>
            <p>{query ? t("tryAnotherKeyword") : t("favoriteHint")}</p>
          </div>
        ) : groups.map((items) => {
          const recording = items[0];
          return (
            <section className="favorite-recording-group" key={recording.recording_id}>
              <div className="favorite-group-heading">
                <div>
                  <span>{recordingDate(recording.recorded_at, locale)}</span>
                  <h2>{recording.recording_title}</h2>
                </div>
                <strong>{t("favoriteCount", { count: items.length })}</strong>
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
                          {t("actual")} {actualTime(favorite.recorded_at, favorite.start_ms, locale)} · {t("elapsed")} {elapsed(favorite.start_ms)}
                        </time>
                      </div>
                      <p className="favorite-sentence-text">{favorite.text}</p>
                      <div className="favorite-note-area">
                        <span className="favorite-note-label">{t("note")}</span>
                        {isEditing ? (
                          <div className="favorite-note-editor">
                            <textarea
                              autoFocus
                              maxLength={2_000}
                              value={drafts[key] ?? ""}
                              placeholder={t("notePlaceholder")}
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
                              >{t("cancel")}</button>
                              <button
                                type="button"
                                disabled={saving === key}
                                onClick={() => void saveNote(favorite)}
                              >{saving === key ? t("saving") : t("saveNote")}</button>
                            </div>
                          </div>
                        ) : (
                          <button
                            className={`favorite-note-preview ${favorite.note ? "has-note" : ""}`}
                            type="button"
                            title={favorite.note || t("addNote")}
                            onClick={() => beginEditing(favorite)}
                          >
                            {favorite.note || t("addNote")}
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

