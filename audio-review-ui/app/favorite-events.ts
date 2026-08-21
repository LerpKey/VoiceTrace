export const FAVORITES_CHANNEL = "audio-review-favorites";

export function announceFavoriteChange() {
  if (typeof BroadcastChannel === "undefined") return;
  const channel = new BroadcastChannel(FAVORITES_CHANNEL);
  channel.postMessage({ type: "favorites-changed" });
  channel.close();
}

