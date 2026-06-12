import { useState, useCallback, useRef } from 'react';
import { Post, InteractionAction } from '../types';
import { FeedAPI } from '../services/api';
import { usePreferences } from '../context/PreferencesContext';

interface UseFeedOptions {
  version?: 'v1' | 'v2' | 'v3';
  limit?: number;
}

interface UseFeedReturn {
  feed: Post[];
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  loadFeed: (refresh?: boolean) => Promise<void>;
  handleInteraction: (post: Post, action: InteractionAction, watchTime?: number) => Promise<void>;
  removePost: (postId: string) => void;
}

export function useFeed({ version = 'v2', limit = 20 }: UseFeedOptions = {}): UseFeedReturn {
  const { state: prefState } = usePreferences();
  const [feed, setFeed] = useState<Post[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const interactedPosts = useRef<Set<string>>(new Set());
  const loadingRef = useRef(false);

  const loadFeed = useCallback(
    async (refresh = false) => {
      if (loadingRef.current && !refresh) return;

      loadingRef.current = true;
      if (refresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);

      try {
        const response = await FeedAPI.getFeed(
          prefState.userId,
          version,
          limit,
          prefState.preferences.mode,
          prefState.preferences.content_type,
        );
        const newPosts = response.data.feed;

        setFeed(refresh ? newPosts : (prev) => {
          const existingIds = new Set(prev.map((p) => p.id));
          const fresh = newPosts.filter((p) => !existingIds.has(p.id));
          return [...prev, ...fresh];
        });
      } catch (err: any) {
        setError(err.message || 'Impossible de charger le feed');
      } finally {
        loadingRef.current = false;
        setLoading(false);
        setRefreshing(false);
      }
    },
    [prefState.userId, prefState.preferences, version, limit]
  );

  const handleInteraction = useCallback(
    async (post: Post, action: InteractionAction, watchTime = 0) => {
      // Optimistic: track locally to avoid double-interaction
      if (interactedPosts.current.has(`${post.id}:${action}`)) return;
      interactedPosts.current.add(`${post.id}:${action}`);

      try {
        await FeedAPI.recordInteraction({
          user_id: prefState.userId,
          post_id: post.id,
          post_text: post.headline || post.text || '',
          action,
          watch_time: watchTime,
        });
      } catch (err) {
        console.error('[Feed] Interaction failed:', err);
        // Remove from tracked so user can retry
        interactedPosts.current.delete(`${post.id}:${action}`);
      }
    },
    [prefState.userId]
  );

  const removePost = useCallback((postId: string) => {
    setFeed((prev) => prev.filter((p) => p.id !== postId));
  }, []);

  return {
    feed,
    loading,
    refreshing,
    error,
    loadFeed,
    handleInteraction,
    removePost,
  };
}