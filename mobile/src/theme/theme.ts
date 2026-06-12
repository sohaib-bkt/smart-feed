export const Colors = {
  primary: '#2563eb',
  primaryLight: '#3b82f6',
  primaryDark: '#1d4ed8',
  secondary: '#64748b',
  success: '#10b981',
  successLight: '#d1fae5',
  danger: '#ef4444',
  dangerLight: '#fee2e2',
  warning: '#f59e0b',
  warningLight: '#fef3c7',
  info: '#06b6d4',
  infoLight: '#cffafe',

  background: '#f8fafc',
  surface: '#ffffff',
  surfaceElevated: '#f1f5f9',
  border: '#e2e8f0',
  borderLight: '#f1f5f9',

  text: '#1e293b',
  textSecondary: '#64748b',
  textLight: '#94a3b8',
  textInverse: '#ffffff',

  // Category colors
  tech: '#6366f1',
  science: '#0891b2',
  sports: '#10b981',
  politics: '#dc2626',
  entertainment: '#f59e0b',
  health: '#ec4899',
  business: '#0d9488',
  other: '#6b7280',

  // Score colors
  scoreHigh: '#10b981',
  scoreMid: '#f59e0b',
  scoreLow: '#ef4444',

  // Toxicity
  toxicityLow: '#10b981',
  toxicityMedium: '#f59e0b',
  toxicityHigh: '#ef4444',
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  '2xl': 24,
  '3xl': 32,
  '4xl': 40,
  '5xl': 48,
};

export const BorderRadius = {
  sm: 6,
  md: 10,
  lg: 14,
  xl: 18,
  '2xl': 24,
  full: 9999,
};

export const Typography = {
  h1: { fontSize: 28, fontWeight: '700' as const, lineHeight: 36 },
  h2: { fontSize: 22, fontWeight: '700' as const, lineHeight: 30 },
  h3: { fontSize: 18, fontWeight: '600' as const, lineHeight: 26 },
  h4: { fontSize: 16, fontWeight: '600' as const, lineHeight: 24 },
  body: { fontSize: 15, fontWeight: '400' as const, lineHeight: 22 },
  bodySmall: { fontSize: 13, fontWeight: '400' as const, lineHeight: 20 },
  caption: { fontSize: 11, fontWeight: '500' as const, lineHeight: 16 },
  label: { fontSize: 12, fontWeight: '600' as const, lineHeight: 16 },
};

export const Shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 1,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 6,
  },
};

export function getCategoryColor(category: string): string {
  const map: Record<string, string> = {
    TECH: Colors.tech,
    TECHNOLOGY: Colors.tech,
    SCIENCE: Colors.science,
    SPORTS: Colors.sports,
    SPORT: Colors.sports,
    POLITICS: Colors.politics,
    ENTERTAINMENT: Colors.entertainment,
    HEALTH: Colors.health,
    BUSINESS: Colors.business,
    'HEALTHY LIVING': Colors.health,
    WELLNESS: Colors.health,
    'STYLE & BEAUTY': '#ec4899',
    'FOOD & DRINK': '#f59e0b',
    TRAVEL: '#06b6d4',
    'HOME & LIVING': '#0d9488',
    'BLACK VOICES': '#6366f1',
    'QUEER VOICES': '#ec4899',
    PARENTING: '#f59e0b',
    PARENTS: '#f59e0b',
    COMEDY: '#f59e0b',
  };
  return map[category?.toUpperCase()] || Colors.other;
}

export function getScoreColor(score: number): string {
  if (score >= 0.7) return Colors.scoreHigh;
  if (score >= 0.4) return Colors.scoreMid;
  return Colors.scoreLow;
}

export function getToxicityColor(score: number): string {
  if (score <= 0.2) return Colors.toxicityLow;
  if (score <= 0.5) return Colors.toxicityMedium;
  return Colors.toxicityHigh;
}