import React, { useEffect, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Text,
} from 'react-native';
import { useFeed } from '../hooks/useFeed';
import PostCard from '../components/PostCard';
import { Post, InteractionAction } from '../types';
import { usePreferences } from '../context/PreferencesContext';
import { Colors, Spacing } from '../theme/theme';

export default function FeedScreen() {
  const {
    feed,
    loading,
    refreshing,
    error,
    loadFeed,
    handleInteraction,
    removePost,
  } = useFeed({ version: 'v3', limit: 20 });

  const { state: prefState } = usePreferences();

  useEffect(() => {
    loadFeed(true);
  }, [prefState.preferences]);

  const handleLoadMore = useCallback(() => {
    if (!loading && feed.length > 0) {
      loadFeed();
    }
  }, [loading, feed.length, loadFeed]);

  const renderItem = useCallback(
    ({ item }: { item: Post }) => (
      <PostCard
        post={item}
        onInteraction={handleInteraction}
        onRemove={removePost}
      />
    ),
    [handleInteraction, removePost]
  );

  const renderFooter = () => {
    if (!loading) return null;
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="small" color={Colors.primary} />
      </View>
    );
  };

  const renderEmpty = () => {
    if (loading) return null;
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyIcon}>📭</Text>
        <Text style={styles.emptyTitle}>Aucun contenu</Text>
        <Text style={styles.emptyText}>
          Modifiez vos préférences pour voir plus de contenu
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <FlatList
        data={feed}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        ListFooterComponent={renderFooter}
        ListEmptyComponent={renderEmpty}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => loadFeed(true)}
            colors={[Colors.primary]}
            tintColor={Colors.primary}
          />
        }
        contentContainerStyle={feed.length === 0 ? styles.emptyContainer : styles.list}
        showsVerticalScrollIndicator={false}
        removeClippedSubviews={true}
        maxToRenderPerBatch={10}
        windowSize={10}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  list: {
    paddingTop: Spacing.md,
    paddingBottom: Spacing['4xl'],
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingTop: Spacing.md,
  },
  empty: {
    alignItems: 'center',
    paddingHorizontal: Spacing['2xl'],
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: Spacing.lg,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: Spacing.sm,
  },
  emptyText: {
    fontSize: 14,
    color: Colors.textLight,
    textAlign: 'center',
  },
  footer: {
    paddingVertical: Spacing.xl,
    alignItems: 'center',
  },
  errorBanner: {
    backgroundColor: Colors.dangerLight,
    padding: Spacing.md,
    marginHorizontal: Spacing.lg,
    marginTop: Spacing.sm,
    borderRadius: 8,
  },
  errorText: {
    color: Colors.danger,
    fontSize: 13,
  },
});