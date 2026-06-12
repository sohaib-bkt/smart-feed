import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Share,
  Alert,
  Animated,
  Vibration,
} from 'react-native';
import { Image } from 'expo-image';
import { Post, InteractionAction } from '../types';
import ExplanationBox from './ExplanationBox';
import VideoPlayer from './VideoPlayer';
import {
  Colors,
  Spacing,
  BorderRadius,
  Typography,
  Shadows,
  getCategoryColor,
  getToxicityColor,
} from '../theme/theme';

interface PostCardProps {
  post: Post;
  onInteraction: (post: Post, action: InteractionAction, watchTime?: number) => Promise<void>;
  onRemove?: (postId: string) => void;
  showExplanation?: boolean;
}

const MODAL_ICONS: Record<string, string> = {
  text: '📝',
  image: '🖼️',
  video: '🎬',
};

export default function PostCard({
  post,
  onInteraction,
  onRemove,
  showExplanation = true,
}: PostCardProps) {
  const [liked, setLiked] = useState(false);
  const [skipped, setSkipped] = useState(false);
  const [showExp, setShowExp] = useState(false);

  const likeScale = useRef(new Animated.Value(1)).current;
  const cardOpacity = useRef(new Animated.Value(1)).current;

  const categoryColor = getCategoryColor(post.category);
  const toxicityColor = getToxicityColor(post.toxicity_score);
  const isHighToxicity = post.toxicity_score > 0.5;

  const animateLike = () => {
    Animated.sequence([
      Animated.spring(likeScale, { toValue: 1.3, useNativeDriver: true, tension: 200 }),
      Animated.spring(likeScale, { toValue: 1, useNativeDriver: true, tension: 200 }),
    ]).start();
  };

  const handleLike = useCallback(async () => {
    if (liked) return;
    setLiked(true);
    animateLike();
    Vibration.vibrate(30);
    await onInteraction(post, 'like');
  }, [liked, post, onInteraction]);

  const handleSkip = useCallback(async () => {
    if (skipped) return;
    setSkipped(true);
    Animated.timing(cardOpacity, {
      toValue: 0.3,
      duration: 300,
      useNativeDriver: true,
    }).start(async () => {
      await onInteraction(post, 'skip');
      onRemove?.(post.id);
    });
  }, [skipped, post, onInteraction, onRemove, cardOpacity]);

  const handleShare = useCallback(async () => {
    await Share.share({
      title: post.headline,
      message: `${post.headline}\n\nVia Smart Feed`,
    });
  }, [post]);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '';
    try {
      return new Date(dateStr).toLocaleDateString('fr-FR', {
        day: 'numeric',
        month: 'short',
      });
    } catch {
      return '';
    }
  };

  return (
    <Animated.View style={[styles.card, { opacity: cardOpacity }]}>
      {/* ── Header ─────────────────────────────────────────────── */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={[styles.categoryBadge, { backgroundColor: categoryColor + '18' }]}>
            <Text style={[styles.categoryText, { color: categoryColor }]}>
              {post.category}
            </Text>
          </View>
          <View style={styles.modalBadge}>
            <Text style={styles.modalIcon}>{MODAL_ICONS[post.modal] || '📄'}</Text>
            <Text style={styles.modalText}>{post.modal}</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          {post.created_at && (
            <Text style={styles.dateText}>{formatDate(post.created_at)}</Text>
          )}
          {isHighToxicity && (
            <View style={styles.toxicityWarning}>
              <Text style={styles.toxicityWarningText}>⚠️</Text>
            </View>
          )}
        </View>
      </View>

      {/* ── Headline ────────────────────────────────────────────── */}
      <View style={styles.content}>
        <Text style={styles.headline} numberOfLines={3}>
          {post.headline || post.text || ''}
        </Text>
        {post.source && (
          <Text style={styles.source}>{post.source}</Text>
        )}
      </View>

      {/* ── Image (for image modality) ───────────────────────────── */}
      {post.modal === 'image' && (
        <View style={styles.mediaContainer}>
          {post.image_url ? (
            <Image
              source={{ uri: post.image_url }}
              style={styles.postImage}
              contentFit="cover"
              transition={300}
              placeholder={Colors.surfaceElevated}
            />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Text style={styles.mediaPlaceholderIcon}>🖼️</Text>
              <Text style={styles.mediaPlaceholderLabel}>Contenu image</Text>
            </View>
          )}
        </View>
      )}

      {/* ── Video (for video modality) ────────────────────────────── */}
      {post.modal === 'video' && (
        <View style={styles.mediaContainer}>
          {post.video_url ? (
            <VideoPlayer uri={post.video_url} thumbnail={post.image_url} />
          ) : (
            <View style={styles.videoPlaceholder}>
              {post.image_url ? (
                <Image
                  source={{ uri: post.image_url }}
                  style={StyleSheet.absoluteFill}
                  contentFit="cover"
                  transition={300}
                />
              ) : (
                <Text style={styles.mediaPlaceholderIcon}>🎬</Text>
              )}
              <View style={styles.videoOverlay} />
              <View style={styles.videoPlayButtonAbsolute}>
                <Text style={styles.videoPlayIcon}>▶</Text>
              </View>
            </View>
          )}
        </View>
      )}

      {/* ── Explanation Box ──────────────────────────────────────── */}
      {showExplanation && post.explanation && (
        <ExplanationBox
          explanation={post.explanation}
          scoreDetail={post.score_detail}
          category={post.category}
        />
      )}

      {/* ── Footer: Actions ─────────────────────────────────────── */}
      <View style={styles.actions}>
        <Animated.View style={{ transform: [{ scale: likeScale }] }}>
          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.likeButton,
              liked && styles.likeButtonActive,
            ]}
            onPress={handleLike}
            activeOpacity={0.8}
          >
            <Text style={styles.actionIcon}>{liked ? '❤️' : '🤍'}</Text>
            <Text style={[styles.actionLabel, liked && styles.likeLabel]}>J'aime</Text>
          </TouchableOpacity>
        </Animated.View>

        <TouchableOpacity
          style={[styles.actionButton, styles.skipButton]}
          onPress={handleSkip}
          activeOpacity={0.8}
        >
          <Text style={styles.actionIcon}>✕</Text>
          <Text style={[styles.actionLabel, styles.skipLabel]}>Ignorer</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.shareButton]}
          onPress={handleShare}
          activeOpacity={0.8}
        >
          <Text style={styles.actionIcon}>↗</Text>
          <Text style={styles.actionLabel}>Partager</Text>
        </TouchableOpacity>
      </View>

      {/* ── Toxicity indicator (subtle) ──────────────────────────── */}
      <View
        style={[
          styles.toxicityBar,
          { backgroundColor: toxicityColor, width: `${post.toxicity_score * 100}%` as any },
        ]}
      />
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.surface,
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    borderRadius: BorderRadius.xl,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: 'hidden',
    ...Shadows.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.md,
    paddingBottom: Spacing.sm,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    flex: 1,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  categoryBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 3,
    borderRadius: BorderRadius.full,
  },
  categoryText: {
    ...Typography.caption,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  modalBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
  modalIcon: {
    fontSize: 11,
  },
  modalText: {
    ...Typography.caption,
    color: Colors.textLight,
    textTransform: 'capitalize',
  },
  dateText: {
    ...Typography.caption,
    color: Colors.textLight,
  },
  toxicityWarning: {
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toxicityWarningText: {
    fontSize: 12,
  },
  content: {
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.md,
  },
  headline: {
    ...Typography.h4,
    color: Colors.text,
    lineHeight: 24,
    marginBottom: Spacing.xs,
  },
  source: {
    ...Typography.caption,
    color: Colors.textLight,
    marginTop: 2,
  },
  mediaContainer: {
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
  },
  postImage: {
    width: '100%',
    height: 200,
    borderRadius: BorderRadius.lg,
  },
  videoPlaceholder: {
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.md,
    height: 180,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  videoPlayButton: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoPlayButtonAbsolute: {
    position: 'absolute',
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: 'rgba(255,255,255,0.9)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  videoPlayIcon: {
    fontSize: 20,
    color: Colors.primary,
    marginLeft: 4,
  },
  imagePlaceholder: {
    width: '100%',
    height: 180,
    backgroundColor: Colors.surfaceElevated,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: BorderRadius.lg,
  },
  mediaPlaceholderIcon: {
    fontSize: 36,
    marginBottom: Spacing.sm,
  },
  mediaPlaceholderLabel: {
    fontSize: 13,
    color: Colors.textLight,
    fontWeight: '500',
  },
  actions: {
    flexDirection: 'row',
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.md,
    gap: Spacing.sm,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.sm + 2,
    borderRadius: BorderRadius.lg,
    gap: Spacing.xs,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  likeButton: {
    backgroundColor: Colors.surface,
  },
  likeButtonActive: {
    backgroundColor: Colors.successLight,
    borderColor: Colors.success,
  },
  skipButton: {
    backgroundColor: Colors.surface,
  },
  shareButton: {
    backgroundColor: Colors.surface,
  },
  actionIcon: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  actionLabel: {
    ...Typography.caption,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  likeLabel: {
    color: Colors.success,
  },
  skipLabel: {
    color: Colors.danger,
  },
  toxicityBar: {
    height: 2,
    position: 'absolute',
    bottom: 0,
    left: 0,
    opacity: 0.5,
  },
});