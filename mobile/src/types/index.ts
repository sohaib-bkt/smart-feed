// Post Types 

export type ModalType = 'text' | 'image' | 'video';
export type InteractionAction = 'like' | 'skip' | 'watch_full' | 'watch_50' | 'comment';
export type ContentType = 'all' | 'text' | 'image' | 'video';
export type RecommendationMode = 'default' | 'focus' | 'fun' | 'learning' | 'fresh';

export interface ScoreDetail {
  content_based?: number;
  collaborative?: number;
  cosine_sim: number;
  recency?: number;
  xgb_score?: number;
}

export interface Explanation {
  summary: string;
  reasons: string[];
  detail: Record<string, any>;
  score: number;
}

export interface Post {
  id: string;
  headline: string;
  text?: string;
  category: string;
  modal: ModalType;
  source?: string;
  score: number;
  score_detail: ScoreDetail;
  explanation: Explanation;
  toxicity_score: number;
  image_url?: string;
  video_url?: string;
  created_at?: string;
  likes?: number;
  views?: number;
}

// User Types 

export interface UserPreferences {
  mode: RecommendationMode;
  interests: string[];
  toxicity_threshold: number;
  content_type: ContentType;
}

export interface UserProfile {
  user_id: string;
  preferences: UserPreferences;
  embedding?: number[];
}

// ─── API Types ────────────────────────────────────────────────────────────────

export interface FeedResponse {
  user_id: string;
  version: string;
  ranking_method: string;
  mode: string;
  count: number;
  feed: Post[];
}

export interface InteractionPayload {
  user_id: string;
  post_id: string;
  post_text?: string;
  action: InteractionAction;
  watch_time?: number;
}

export interface SearchResult {
  posts: Post[];
  query: string;
  count: number;
}

// ─── Navigation Types ─────────────────────────────────────────────────────────

export type RootTabParamList = {
  Feed: undefined;
  Search: undefined;
  Profile: undefined;
  Settings: undefined;
};

export type FeedStackParamList = {
  FeedMain: undefined;
  PostDetail: { post: Post };
};