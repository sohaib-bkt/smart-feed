import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { Explanation, ScoreDetail } from '../types';
import { Colors, Spacing, BorderRadius, Typography, getScoreColor, Shadows } from '../theme/theme';

interface ExplanationBoxProps {
  explanation: Explanation;
  scoreDetail: ScoreDetail;
  category: string;
}

export default function ExplanationBox({ explanation, scoreDetail, category }: ExplanationBoxProps) {
  const [expanded, setExpanded] = useState(false);
  const animHeight = useRef(new Animated.Value(0)).current;
  const animOpacity = useRef(new Animated.Value(0)).current;

  const toggle = () => {
    const toExpand = !expanded;
    setExpanded(toExpand);
    Animated.parallel([
      Animated.spring(animHeight, {
        toValue: toExpand ? 1 : 0,
        useNativeDriver: false,
        tension: 80,
        friction: 10,
      }),
      Animated.timing(animOpacity, {
        toValue: toExpand ? 1 : 0,
        duration: 200,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const scoreColor = getScoreColor(explanation.score);
  const scorePercent = Math.round(explanation.score * 100);

  const metrics = [
    { label: 'Similarité', value: scoreDetail.cosine_sim, key: 'cosine_sim' },
    { label: 'Collaboratif', value: scoreDetail.collaborative, key: 'collaborative' },
    { label: 'Content-Based', value: scoreDetail.content_based, key: 'content_based' },
    { label: 'Récence', value: scoreDetail.recency, key: 'recency' },
  ].filter((m) => m.value !== undefined && m.value !== null);

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.header} onPress={toggle} activeOpacity={0.7}>
        <View style={styles.headerLeft}>
          <Text style={styles.icon}>🔍</Text>
          <Text style={styles.headerText}>Pourquoi ce post ?</Text>
        </View>
        <View style={styles.headerRight}>
          <View style={[styles.scoreBadge, { backgroundColor: scoreColor + '20' }]}>
            <Text style={[styles.scoreText, { color: scoreColor }]}>{scorePercent}%</Text>
          </View>
          <Text style={[styles.chevron, expanded && styles.chevronOpen]}>›</Text>
        </View>
      </TouchableOpacity>

      <Animated.View
        style={[
          styles.body,
          {
            maxHeight: animHeight.interpolate({
              inputRange: [0, 1],
              outputRange: [0, 400],
            }),
            opacity: animOpacity,
          },
        ]}
      >
        <ScrollView scrollEnabled={false} style={styles.bodyScroll}>
          {/* Summary */}
          <View style={styles.summaryBox}>
            <Text style={styles.summaryText}>{explanation.summary}</Text>
          </View>

          {/* Reasons */}
          {explanation.reasons.length > 0 && (
            <View style={styles.reasonsSection}>
              <Text style={styles.sectionTitle}>Raisons</Text>
              {explanation.reasons.map((reason, idx) => (
                <View key={idx} style={styles.reasonRow}>
                  <View style={styles.reasonDot} />
                  <Text style={styles.reasonText}>{reason}</Text>
                </View>
              ))}
            </View>
          )}

          {/* Score breakdown */}
          {metrics.length > 0 && (
            <View style={styles.metricsSection}>
              <Text style={styles.sectionTitle}>Détails des scores</Text>
              {metrics.map((metric) => (
                <View key={metric.key} style={styles.metricRow}>
                  <Text style={styles.metricLabel}>{metric.label}</Text>
                  <View style={styles.metricBarContainer}>
                    <View
                      style={[
                        styles.metricBar,
                        {
                          width: `${Math.round((metric.value ?? 0) * 100)}%`,
                          backgroundColor: getScoreColor(metric.value ?? 0),
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.metricValue}>
                    {Math.round((metric.value ?? 0) * 100)}%
                  </Text>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: 'hidden',
    ...Shadows.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm + 2,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  icon: {
    fontSize: 14,
  },
  headerText: {
    ...Typography.label,
    color: Colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  scoreBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: BorderRadius.full,
  },
  scoreText: {
    ...Typography.label,
    fontWeight: '700',
  },
  chevron: {
    fontSize: 20,
    color: Colors.textLight,
    fontWeight: '300',
    transform: [{ rotate: '0deg' }],
  },
  chevronOpen: {
    transform: [{ rotate: '90deg' }],
  },
  body: {
    overflow: 'hidden',
  },
  bodyScroll: {
    paddingHorizontal: Spacing.md,
    paddingBottom: Spacing.md,
  },
  summaryBox: {
    backgroundColor: Colors.primary + '10',
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    marginBottom: Spacing.md,
    borderLeftWidth: 3,
    borderLeftColor: Colors.primary,
  },
  summaryText: {
    ...Typography.bodySmall,
    color: Colors.text,
    fontStyle: 'italic',
    lineHeight: 20,
  },
  sectionTitle: {
    ...Typography.caption,
    color: Colors.textLight,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: Spacing.sm,
  },
  reasonsSection: {
    marginBottom: Spacing.md,
  },
  reasonRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: Spacing.xs,
    gap: Spacing.sm,
  },
  reasonDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.primary,
    marginTop: 6,
  },
  reasonText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    flex: 1,
  },
  metricsSection: {
    marginBottom: Spacing.sm,
  },
  metricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.xs + 2,
    gap: Spacing.sm,
  },
  metricLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    width: 90,
  },
  metricBarContainer: {
    flex: 1,
    height: 5,
    backgroundColor: Colors.border,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  metricBar: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  metricValue: {
    ...Typography.caption,
    color: Colors.textSecondary,
    width: 32,
    textAlign: 'right',
  },
});