import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserPreferences, RecommendationMode, ContentType } from '../types';
import { PreferencesAPI } from '../services/api';

// Types

interface PreferencesState {
  userId: string;
  preferences: UserPreferences;
  loading: boolean;
  error: string | null;
}

type PreferencesAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_PREFERENCES'; payload: UserPreferences }
  | { type: 'SET_USER'; payload: string }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'UPDATE_MODE'; payload: RecommendationMode }
  | { type: 'UPDATE_TOXICITY'; payload: number }
  | { type: 'UPDATE_CONTENT_TYPE'; payload: ContentType };

interface PreferencesContextValue {
  state: PreferencesState;
  setUser: (userId: string) => void;
  updateMode: (mode: RecommendationMode) => Promise<void>;
  updateToxicityThreshold: (threshold: number) => Promise<void>;
  updateContentType: (contentType: ContentType) => Promise<void>;
  savePreferences: (prefs: UserPreferences) => Promise<void>;
  refreshPreferences: () => Promise<void>;
}

// Initial State 

const DEFAULT_PREFERENCES: UserPreferences = {
  mode: 'default',
  interests: [],
  toxicity_threshold: 0.3,
  content_type: 'all',
};

const STORAGE_KEY = '@smartfeed:preferences';
const USER_KEY = '@smartfeed:userId';

// ─── Reducer ──────────────────────────────────────────────────────────────────

function reducer(state: PreferencesState, action: PreferencesAction): PreferencesState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_USER':
      return { ...state, userId: action.payload };
    case 'SET_PREFERENCES':
      return { ...state, preferences: action.payload, loading: false, error: null };
    case 'UPDATE_MODE':
      return {
        ...state,
        preferences: { ...state.preferences, mode: action.payload },
      };
    case 'UPDATE_TOXICITY':
      return {
        ...state,
        preferences: { ...state.preferences, toxicity_threshold: action.payload },
      };
    case 'UPDATE_CONTENT_TYPE':
      return {
        ...state,
        preferences: { ...state.preferences, content_type: action.payload },
      };
    default:
      return state;
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

// ─── Provider ─────────────────────────────────────────────────────────────────

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, {
    userId: 'user_demo',
    preferences: DEFAULT_PREFERENCES,
    loading: false,
    error: null,
  });

  // Load persisted preferences on mount
  useEffect(() => {
    loadPersistedData();
  }, []);

  // Sync to AsyncStorage whenever preferences change
  useEffect(() => {
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state.preferences)).catch(console.error);
  }, [state.preferences]);

  const loadPersistedData = async () => {
    try {
      const [savedPrefs, savedUserId] = await Promise.all([
        AsyncStorage.getItem(STORAGE_KEY),
        AsyncStorage.getItem(USER_KEY),
      ]);

      if (savedUserId) {
        dispatch({ type: 'SET_USER', payload: savedUserId });
      }

      if (savedPrefs) {
        dispatch({ type: 'SET_PREFERENCES', payload: JSON.parse(savedPrefs) });
      }
    } catch (err) {
      console.error('Failed to load persisted preferences:', err);
    }
  };

  const setUser = useCallback(async (userId: string) => {
    dispatch({ type: 'SET_USER', payload: userId });
    await AsyncStorage.setItem(USER_KEY, userId);
  }, []);

  const refreshPreferences = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await PreferencesAPI.getPreferences(state.userId);
      dispatch({ type: 'SET_PREFERENCES', payload: response.data });
    } catch (err) {
      // Fallback to cached preferences silently
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [state.userId]);

  const savePreferences = useCallback(
    async (prefs: UserPreferences) => {
      try {
        await PreferencesAPI.updatePreferences(state.userId, prefs);
        dispatch({ type: 'SET_PREFERENCES', payload: prefs });
      } catch (err) {
        // Still update local state even if API fails
        dispatch({ type: 'SET_PREFERENCES', payload: prefs });
      }
    },
    [state.userId]
  );

  const updateMode = useCallback(
    async (mode: RecommendationMode) => {
      dispatch({ type: 'UPDATE_MODE', payload: mode });
      try {
        await PreferencesAPI.setMode(state.userId, mode);
      } catch (err) {
        console.error('Failed to update mode:', err);
      }
    },
    [state.userId]
  );

  const updateToxicityThreshold = useCallback(
    async (threshold: number) => {
      dispatch({ type: 'UPDATE_TOXICITY', payload: threshold });
      try {
        const updated = { ...state.preferences, toxicity_threshold: threshold };
        await PreferencesAPI.updatePreferences(state.userId, updated);
      } catch (err) {
        console.error('Failed to update toxicity threshold:', err);
      }
    },
    [state.userId, state.preferences]
  );

  const updateContentType = useCallback(
    async (contentType: ContentType) => {
      dispatch({ type: 'UPDATE_CONTENT_TYPE', payload: contentType });
      try {
        const updated = { ...state.preferences, content_type: contentType };
        await PreferencesAPI.updatePreferences(state.userId, updated);
      } catch (err) {
        console.error('Failed to update content type:', err);
      }
    },
    [state.userId, state.preferences]
  );

  return (
    <PreferencesContext.Provider
      value={{
        state,
        setUser,
        updateMode,
        updateToxicityThreshold,
        updateContentType,
        savePreferences,
        refreshPreferences,
      }}
    >
      {children}
    </PreferencesContext.Provider>
  );
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function usePreferences() {
  const ctx = useContext(PreferencesContext);
  if (!ctx) {
    throw new Error('usePreferences must be used within a PreferencesProvider');
  }
  return ctx;
}