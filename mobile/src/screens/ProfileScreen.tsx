// src/screens/ProfileScreen.tsx
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { usePreferences } from '../context/PreferencesContext';
import {
  Colors,
  Spacing,
  BorderRadius,
  Typography,
  Shadows,
} from '../theme/theme';

interface StatCardProps {
  label: string;
  value: string | number;
  icon: string;
  color?: string;
}

function StatCard({ label, value, icon, color = Colors.primary }: StatCardProps) {
  return (
    <View style={styles.statCard}>
      <Text style={styles.statIcon}>{icon}</Text>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

export default function ProfileScreen() {
  const { state } = usePreferences();
  const { preferences, userId } = state;

  // Données mockées pour la démo
  const stats = {
    postsViewed: 342,
    likes: 89,
    skips: 127,
    sessionsCount: 23,
    avgSession: '14 min',
  };

  const modeLabels: Record<string, string> = {
    default: 'Standard',
    focus: 'Focus',
    fun: 'Fun',
    learning: 'Apprentissage',
    fresh: 'Découverte',
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Profile Header ──────────────────────────────── */}
      <View style={styles.header}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>
            {userId?.charAt(0)?.toUpperCase() || 'U'}
          </Text>
        </View>
        <Text style={styles.userId}>@{userId}</Text>
        <View style={styles.modeBadge}>
          <Text style={styles.modeText}>
            Mode : {modeLabels[preferences.mode] || preferences.mode}
          </Text>
        </View>
      </View>

      {/* ── Statistics Grid ──────────────────────────────── */}
      <Text style={styles.sectionTitle}>Statistiques</Text>
      <View style={styles.statsGrid}>
        <StatCard icon="👁️" value={stats.postsViewed} label="Posts vus" />
        <StatCard icon="❤️" value={stats.likes} label="Likes" color={Colors.success} />
        <StatCard icon="✕" value={stats.skips} label="Ignorés" color={Colors.danger} />
        <StatCard icon="📅" value={stats.sessionsCount} label="Sessions" color={Colors.info} />
      </View>

      {/* ── Content Type Distribution ────────────────────── */}
      <Text style={styles.sectionTitle}>Distribution du contenu</Text>
      <View style={styles.distributionCard}>
        <View style={styles.distRow}>
          <Text style={styles.distLabel}>📝 Texte</Text>
          <View style={styles.distBarContainer}>
            <View style={[styles.distBar, { width: '45%', backgroundColor: Colors.primary }]} />
          </View>
          <Text style={styles.distValue}>45%</Text>
        </View>
        <View style={styles.distRow}>
          <Text style={styles.distLabel}>🖼️ Images</Text>
          <View style={styles.distBarContainer}>
            <View style={[styles.distBar, { width: '30%', backgroundColor: Colors.success }]} />
          </View>
          <Text style={styles.distValue}>30%</Text>
        </View>
        <View style={styles.distRow}>
          <Text style={styles.distLabel}>🎬 Vidéos</Text>
          <View style={styles.distBarContainer}>
            <View style={[styles.distBar, { width: '25%', backgroundColor: Colors.warning }]} />
          </View>
          <Text style={styles.distValue}>25%</Text>
        </View>
      </View>

      {/* ── Preferences Summary ──────────────────────────── */}
      <Text style={styles.sectionTitle}>Préférences</Text>
      <View style={styles.prefCard}>
        <View style={styles.prefRow}>
          <Text style={styles.prefLabel}>Type de contenu</Text>
          <Text style={styles.prefValue}>{preferences.content_type}</Text>
        </View>
        <View style={styles.prefRow}>
          <Text style={styles.prefLabel}>Seuil toxicité</Text>
          <Text style={styles.prefValue}>{Math.round(preferences.toxicity_threshold * 100)}%</Text>
        </View>
        <View style={styles.prefRow}>
          <Text style={styles.prefLabel}>Centres d'intérêt</Text>
          <Text style={styles.prefValue}>
            {preferences.interests.length > 0
              ? preferences.interests.join(', ')
              : 'Aucun défini'}
          </Text>
        </View>
      </View>

      {/* ── Session Info ─────────────────────────────────── */}
      <View style={styles.sessionCard}>
        <Text style={styles.sessionIcon}>⏱️</Text>
        <View style={styles.sessionInfo}>
          <Text style={styles.sessionTitle}>Temps moyen par session</Text>
          <Text style={styles.sessionValue}>{stats.avgSession}</Text>
        </View>
      </View>

      {/* ─── Espace en bas ───────────────────────────────── */}
      <View style={{ height: Spacing['4xl'] }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    padding: Spacing.lg,
  },

  // ── Header ──────────────────────────────────
  header: {
    alignItems: 'center',
    paddingVertical: Spacing['2xl'],
    marginBottom: Spacing.lg,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.md,
    ...Shadows.md,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: '700',
    color: Colors.textInverse,
  },
  userId: {
    ...Typography.h2,
    color: Colors.text,
    marginBottom: Spacing.sm,
  },
  modeBadge: {
    backgroundColor: Colors.primary + '18',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  modeText: {
    ...Typography.caption,
    color: Colors.primary,
    fontWeight: '600',
  },

  // ── Section Titles ─────────────────────────
  sectionTitle: {
    ...Typography.h3,
    color: Colors.text,
    marginBottom: Spacing.md,
    marginTop: Spacing.lg,
  },

  // ── Stats Grid ─────────────────────────────
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  statCard: {
    width: '47%',
    backgroundColor: Colors.surface,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
    ...Shadows.sm,
  },
  statIcon: {
    fontSize: 24,
    marginBottom: Spacing.sm,
  },
  statValue: {
    ...Typography.h2,
    color: Colors.primary,
    marginBottom: 2,
  },
  statLabel: {
    ...Typography.caption,
    color: Colors.textLight,
  },

  // ── Distribution ────────────────────────────
  distributionCard: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.border,
    gap: Spacing.md,
    ...Shadows.sm,
  },
  distRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  distLabel: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    width: 80,
  },
  distBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: Colors.border,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  distBar: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  distValue: {
    ...Typography.caption,
    color: Colors.textSecondary,
    width: 35,
    textAlign: 'right',
  },

  // ── Preferences ─────────────────────────────
  prefCard: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
    borderWidth: 1,
    borderColor: Colors.border,
    gap: Spacing.md,
    ...Shadows.sm,
  },
  prefRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  prefLabel: {
    ...Typography.bodySmall,
    color: Colors.textLight,
  },
  prefValue: {
    ...Typography.bodySmall,
    color: Colors.text,
    fontWeight: '600',
    textTransform: 'capitalize',
  },

  // ── Session ─────────────────────────────────
  sessionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
    marginTop: Spacing.md,
    borderWidth: 1,
    borderColor: Colors.border,
    gap: Spacing.lg,
    ...Shadows.sm,
  },
  sessionIcon: {
    fontSize: 28,
  },
  sessionInfo: {
    flex: 1,
  },
  sessionTitle: {
    ...Typography.caption,
    color: Colors.textLight,
  },
  sessionValue: {
    ...Typography.h3,
    color: Colors.text,
  },
});