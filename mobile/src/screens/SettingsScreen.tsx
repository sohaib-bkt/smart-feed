import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { usePreferences } from '../context/PreferencesContext';
import { RecommendationMode, ContentType } from '../types';
import { Colors, Spacing, BorderRadius, Typography, Shadows } from '../theme/theme';

const MODE_CATEGORIES: Record<RecommendationMode, { label: string; icon: string; desc: string; categories: string[] }> = {
  default: {
    label: 'Standard',
    icon: '📰',
    desc: 'Toutes les catégories mélangées',
    categories: ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'COMEDY', 'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS'],
  },
  focus: {
    label: 'Focus',
    icon: '🎯',
    desc: 'Contenu sérieux et approfondi',
    categories: ['POLITICS', 'BUSINESS', 'HOME & LIVING', 'BLACK VOICES', 'QUEER VOICES', 'PARENTS'],
  },
  fun: {
    label: 'Fun',
    icon: '🎉',
    desc: 'Divertissement et détente',
    categories: ['COMEDY', 'ENTERTAINMENT', 'SPORTS', 'FOOD & DRINK', 'TRAVEL', 'STYLE & BEAUTY'],
  },
  learning: {
    label: 'Apprentissage',
    icon: '📚',
    desc: 'Bien-être et développement',
    categories: ['WELLNESS', 'HEALTHY LIVING', 'PARENTING'],
  },
  fresh: {
    label: 'Découverte',
    icon: '🆕',
    desc: 'Tout le contenu récent',
    categories: ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'COMEDY', 'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS'],
  },
};

export default function SettingsScreen() {
  const { state, updateMode, updateContentType } = usePreferences();
  const { preferences } = state;
  const activeCategories = MODE_CATEGORIES[preferences.mode]?.categories ?? [];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>Mode de navigation</Text>
      <View style={styles.modesGrid}>
        {(Object.keys(MODE_CATEGORIES) as RecommendationMode[]).map((key) => {
          const mode = MODE_CATEGORIES[key];
          const isActive = preferences.mode === key;
          return (
            <TouchableOpacity
              key={key}
              style={[
                styles.modeCard,
                isActive && styles.modeCardActive,
              ]}
              onPress={() => updateMode(key)}
              activeOpacity={0.7}
            >
              <Text style={styles.modeIcon}>{mode.icon}</Text>
              <Text style={[styles.modeLabel, isActive && styles.modeLabelActive]}>
                {mode.label}
              </Text>
              <Text style={styles.modeDesc}>{mode.desc}</Text>
              {isActive && (
                <View style={styles.activeCategories}>
                  {activeCategories.map((cat) => (
                    <View key={cat} style={styles.activeChip}>
                      <Text style={styles.activeChipText}>{cat}</Text>
                    </View>
                  ))}
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      <Text style={styles.sectionTitle}>Type de contenu</Text>
      <View style={styles.chipsContainer}>
        {(['all', 'text', 'image', 'video'] as ContentType[]).map((type) => (
          <TouchableOpacity
            key={type}
            style={[
              styles.chip,
              preferences.content_type === type && styles.chipActive,
            ]}
            onPress={() => updateContentType(type)}
          >
            <Text
              style={[
                styles.chipText,
                preferences.content_type === type && styles.chipTextActive,
              ]}
            >
              {type === 'all' ? 'Tout' : type}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
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
    paddingBottom: Spacing['5xl'],
  },
  sectionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.lg,
    marginTop: Spacing.xl,
  },
  modesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.md,
  },
  modeCard: {
    width: '47%',
    backgroundColor: Colors.surface,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    borderWidth: 2,
    borderColor: Colors.border,
    ...Shadows.sm,
  },
  modeCardActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primary + '08',
  },
  modeIcon: {
    fontSize: 24,
    marginBottom: Spacing.sm,
  },
  modeLabel: {
    ...Typography.h4,
    marginBottom: Spacing.xs,
  },
  modeLabelActive: {
    color: Colors.primary,
  },
  modeDesc: {
    ...Typography.caption,
    color: Colors.textLight,
    marginBottom: Spacing.sm,
  },
  activeCategories: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: Spacing.xs,
  },
  activeChip: {
    backgroundColor: Colors.primary + '15',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: BorderRadius.sm,
  },
  activeChipText: {
    fontSize: 9,
    color: Colors.primary,
    fontWeight: '600',
  },
  chipsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  chip: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.full,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  chipActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primary + '15',
  },
  chipText: {
    ...Typography.label,
    color: Colors.textSecondary,
    textTransform: 'capitalize',
  },
  chipTextActive: {
    color: Colors.primary,
  },
});