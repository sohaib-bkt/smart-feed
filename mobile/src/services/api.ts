import axios, { AxiosInstance } from 'axios';
import {
  FeedResponse,
  InteractionPayload,
  UserPreferences,
  Post,
  RecommendationMode,
} from '../types';

// Configuration 
const API_BASE = 'http://localhost:8000/api';
// pour un appareil physique : 'http://192.168.X.X:8000/api'

const client: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptors 
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Erreur réseau';
    console.error('[API Error]', message, error.config?.url);
    return Promise.reject(new Error(message));
  }
);

// Feed API 
export const FeedAPI = {
  getFeed: (userId: string, version: 'v1' | 'v2' | 'v3' = 'v2', limit = 20) =>
    client.get<FeedResponse>(`/feed/${userId}`, {
      params: { version, limit },
    }),

  recordInteraction: (payload: InteractionPayload) =>
    client.post('/interact', payload),
};

// Preferences API
export const PreferencesAPI = {
  getPreferences: (userId: string) =>
    client.get<UserPreferences>(`/preferences/${userId}`),

  updatePreferences: (userId: string, prefs: UserPreferences) =>
    client.put(`/preferences/${userId}`, prefs),

  setMode: (userId: string, mode: RecommendationMode) =>
    client.post(`/preferences/${userId}/mode/${mode}`),
};

// Search API 
export const SearchAPI = {
  search: (query: string, userId: string, limit = 10) =>
    client.post('/recommend/search', { query, user_id: userId, limit }),

  searchByImage: (imageBase64: string, userId: string, limit = 10) =>
    client.post('/recommend/search', {
      image_base64: imageBase64,
      user_id: userId,
      limit,
    }),
};

export default client;