import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
} from 'react-native';
import { usePreferences } from '../context/PreferencesContext';
import { RecommendationMode, ContentType } from '../types';
import { Colors, Spacing, BorderRadius, Typography, Shadows } from '../theme/theme';

const MODES: { value: RecommendationMode; label: string; icon: string; desc: string }[] = [
  { value: 'default', label: 'Standard', icon: '📰', desc: 'Recommandations équilibrées' },
  { value: 'focus', label: 'Focus', icon: '🎯', desc: 'Contenu approfondi' },
  { value: 'fun', label: 'Fun', icon: '🎉', desc: 'Divertissement léger' },
  { value: 'learning', label: 'Apprentissage', icon: '📚', desc: 'Contenu éducatif' },
  { value: 'fresh', label: 'Découverte', icon: '🆕', desc: 'Nouveautés et diversité' },
];

export default function SettingsScreen() {
  const { state, updateMode, updateContentType } = usePreferences();
  const { preferences } = state;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>Mode de recommandation</Text>
      <View style={styles.modesGrid}>
        {MODES.map((mode) => (
          <TouchableOpacity
            key={mode.value}
            style={[
              styles.modeCard,
              preferences.mode === mode.value && styles.modeCardActive,
            ]}
            onPress={() => updateMode(mode.value)}
            activeOpacity={0.7}
          >
            <Text style={styles.modeIcon}>{mode.icon}</Text>
            <Text
              style={[
                styles.modeLabel,
                preferences.mode === mode.value && styles.modeLabelActive,
              ]}
            >
              {mode.label}
            </Text>
            <Text style={styles.modeDesc}>{mode.desc}</Text>
          </TouchableOpacity>
        ))}
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