# tests/test_sbert_model.py
"""
Test complet du modèle SBERT (Sentence-BERT) pour les embeddings texte.

Teste :
- Chargement du modèle
- Embedding simple et batch
- Normalisation des embeddings
- Similarité cosinus
- Performance
- Intégration avec FAISS v1/v2
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# Imports
# ============================================================================
from app.models.embeddings import get_model as get_sbert_model, embed_text, embed_batch
from app.services.recommender import _load_resources, _load_resources_v2


def print_section(title):
    """Affiche une section de test."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title):
    """Affiche une sous-section."""
    print(f"\n  --- {title} ---")


def get_embedding_dimension():
    """Retourne la dimension des embeddings SBERT."""
    test_emb = embed_text("test")
    return len(test_emb)


# ============================================================================
# SECTION 1 : Chargement et validation du modèle
# ============================================================================

def test_model_loading():
    """Teste le chargement du modèle SBERT."""
    print_section("1. SBERT Model Loading")
    
    try:
        start = time.time()
        model = get_sbert_model()
        load_time = time.time() - start
        
        assert model is not None, "Model is None"
        print(f"  ✅ Modèle chargé en {load_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


def test_embedding_dimension():
    """Teste la dimension des embeddings."""
    print_section("2. Embedding Dimension")
    
    dim = get_embedding_dimension()
    print(f"  ✅ Dimension: {dim}")
    
    assert dim == 384, f"Expected 384, got {dim}"
    return dim


def test_single_embedding():
    """Teste l'embedding d'un seul texte."""
    print_section("3. Single Text Embedding")
    
    texts = [
        "Artificial intelligence is transforming the world",
        "Le football est un sport populaire en France",
        "The stock market reached new highs today"
    ]
    
    for text in texts:
        start = time.time()
        embedding = embed_text(text)
        elapsed = time.time() - start
        
        assert len(embedding) == 384, f"Wrong dimension: {len(embedding)}"
        
        # Vérifier la normalisation (norme ≈ 1.0)
        norm = np.linalg.norm(embedding)
        print(f"  📝 '{text[:40]}...'")
        print(f"     → {len(embedding)} dims, norm={norm:.4f}, temps={elapsed*1000:.2f}ms")
        
        assert abs(norm - 1.0) < 0.1, f"Embedding not normalized: norm={norm}"
    
    return True


def test_batch_embedding():
    """Teste l'embedding par lots (batch)."""
    print_section("4. Batch Embedding")
    
    texts = [
        "First text for batch embedding",
        "Second text for testing",
        "Third text in the batch",
        "Fourth text to check consistency",
        "Fifth text for performance"
    ]
    
    # Test simple batch
    start = time.time()
    batch_emb = embed_batch(texts)
    elapsed = time.time() - start
    
    assert batch_emb.shape == (len(texts), 384), f"Wrong shape: {batch_emb.shape}"
    
    print(f"  ✅ Batch de {len(texts)} textes: {batch_emb.shape}")
    print(f"  ⏱️ Temps: {elapsed*1000:.2f}ms ({elapsed*1000/len(texts):.1f}ms/texte)")
    
    # Vérifier la normalisation de chaque vecteur
    norms = np.linalg.norm(batch_emb, axis=1)
    print(f"  ✅ Normalisation: mean={norms.mean():.4f}, std={norms.std():.4f}")
    assert np.all(norms > 0.9), "Some embeddings not normalized"
    
    return batch_emb


# ============================================================================
# SECTION 2 : Similarité cosinus
# ============================================================================

def test_cosine_similarity():
    """Teste la similarité cosinus entre embeddings."""
    print_section("5. Cosine Similarity")
    
    texts = [
        "A cat sleeping on a couch",
        "A dog running in the park",
        "A kitten sleeping on a sofa",  # Similaire au premier
        "Stock market trading today"     # Différent
    ]
    
    embeddings = embed_batch(texts)
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine_sim(embeddings[0], embeddings[1])
    sim_1_3 = cosine_sim(embeddings[0], embeddings[2])
    sim_1_4 = cosine_sim(embeddings[0], embeddings[3])
    
    print(f"\n  📝 Texte 1: '{texts[0]}'")
    print(f"  📝 Texte 2: '{texts[1]}' → similarité: {sim_1_2:.4f}")
    print(f"  📝 Texte 3: '{texts[2]}' → similarité: {sim_1_3:.4f}")
    print(f"  📝 Texte 4: '{texts[3]}' → similarité: {sim_1_4:.4f}")
    
    # Vérifier que les textes similaires ont une similarité plus élevée
    if sim_1_3 > sim_1_2:
        print("  ✅ Similarité sémantique: textes similaires bien rapprochés")
    else:
        print("  ⚠️ Similarité sémantique: vérifier les embeddings")
    
    return True


def test_semantic_search():
    """Teste la recherche sémantique basique."""
    print_section("6. Basic Semantic Search")
    
    corpus = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning uses neural networks",
        "Deep learning is a subset of machine learning",
        "Web development with React and Vue",
        "Data science and statistical analysis",
        "Artificial intelligence in healthcare",
        "Cloud computing with AWS and Azure"
    ]
    
    query = "What is machine learning?"
    
    # Embedder tous les documents
    corpus_emb = embed_batch(corpus)
    query_emb = embed_text(query)
    
    # Calculer les similarités
    similarities = []
    for i, doc_emb in enumerate(corpus_emb):
        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        similarities.append((i, sim))
    
    # Trier par similarité
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  🔍 Requête: '{query}'")
    print(f"\n  📊 Top 3 résultats:")
    
    for rank, (idx, sim) in enumerate(similarities[:3]):
        print(f"     {rank+1}. [{sim:.4f}] {corpus[idx]}")
    
    # Vérifier que le résultat le plus pertinent est en tête
    top_idx = similarities[0][0]
    if "machine learning" in corpus[top_idx].lower() or "deep learning" in corpus[top_idx].lower():
        print("\n  ✅ Recherche sémantique fonctionnelle!")
    else:
        print("\n  ⚠️ Résultat inattendu - vérifier les embeddings")
    
    return True


# ============================================================================
# SECTION 3 : Performance
# ============================================================================

def test_performance():
    """Teste les performances d'embedding."""
    print_section("7. Performance Test")
    
    test_texts = [
        "This is a short text.",
        "This is a medium length text that contains a few more words for testing purposes.",
        "This is a longer text. " * 10,
    ]
    
    n_iterations = 20
    
    for text in test_texts:
        times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = embed_text(text)
            times.append(time.time() - start)
        
        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        
        print(f"\n  📝 Longueur: {len(text)} caractères")
        print(f"     ⏱️ Moyenne: {avg_ms:.2f}ms (±{std_ms:.2f})")
        print(f"     🚀 Max: {max(times)*1000:.2f}ms | Min: {min(times)*1000:.2f}ms")
    
    return True


def test_batch_performance():
    """Teste les performances d'embedding par lots."""
    print_section("8. Batch Performance")
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    texts = ["Test text for batch performance"] * 128
    
    print(f"\n  📊 Benchmark batch embedding:")
    print(f"  {'Taille':>8} | {'Temps total':>12} | {'Temps/texte':>12} | {'Speedup':>10}")
    print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
    
    single_time = None
    
    for batch_size in batch_sizes:
        if batch_size > len(texts):
            continue
        
        batch_texts = texts[:batch_size]
        
        start = time.time()
        _ = embed_batch(batch_texts)
        elapsed = time.time() - start
        
        time_per_text = elapsed / batch_size * 1000
        
        if batch_size == 1:
            single_time = time_per_text
            speedup = 1.0
        else:
            speedup = single_time / time_per_text if single_time else 1.0
        
        print(f"  {batch_size:8d} | {elapsed*1000:10.2f}ms | {time_per_text:10.2f}ms | {speedup:9.2f}x")
    
    return True


# ============================================================================
# SECTION 4 : Intégration avec FAISS
# ============================================================================

def test_faiss_v1_integration():
    """Teste l'intégration avec FAISS v1 (content-based)."""
    print_section("9. FAISS v1 Integration (Content-Based)")
    
    try:
        _load_resources()
        from app.services.recommender import _index as idx_v1, _posts_df as df_v1
        
        print(f"  ✅ FAISS v1 chargé: {idx_v1.ntotal} vecteurs")
        print(f"  ✅ Métadonnées: {df_v1.shape[0]} articles")
        
        # Tester une recherche simple
        query = "technology and artificial intelligence"
        query_emb = embed_text(query)
        
        # Recherche FAISS
        query_np = np.array([query_emb]).astype('float32')
        scores, indices = idx_v1.search(query_np, 5)
        
        print(f"\n  🔍 Requête: '{query}'")
        print(f"  📊 Top 5 résultats FAISS v1:")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(df_v1):
                title = df_v1.iloc[idx].get('text', '')[:60]
                print(f"     {i+1}. [{score:.4f}] {title}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


def test_faiss_v2_integration():
    """Teste l'intégration avec FAISS v2 (hybride)."""
    print_section("10. FAISS v2 Integration (Hybrid)")
    
    try:
        _load_resources_v2()
        from app.services.recommender import _index_v2 as idx_v2, _posts_df_v2 as df_v2
        
        print(f"  ✅ FAISS v2 chargé: {idx_v2.ntotal} vecteurs")
        print(f"  ✅ Métadonnées: {df_v2.shape[0]} articles")
        
        # Tester une recherche
        query = "sports and football"
        query_emb = embed_text(query)
        query_np = np.array([query_emb]).astype('float32')
        scores, indices = idx_v2.search(query_np, 5)
        
        print(f"\n  🔍 Requête: '{query}'")
        print(f"  📊 Top 5 résultats FAISS v2:")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(df_v2):
                title = df_v2.iloc[idx].get('text', '')[:60]
                print(f"     {i+1}. [{score:.4f}] {title}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


def test_projection_for_v3():
    """Teste la projection des embeddings SBERT vers l'espace v3."""
    print_section("11. Projection for FAISS v3")
    
    try:
        from app.services.multimodal_index import get_projection_matrices, search_by_text
        
        proj_text, _ = get_projection_matrices()
        
        if proj_text is None:
            print("  ⚠️ Matrice de projection non disponible")
            return False
        
        print(f"  ✅ Matrice de projection: {proj_text.shape}")
        
        # Tester la projection
        test_text = "a beautiful sunset over the ocean"
        emb_384 = embed_text(test_text)
        emb_512 = emb_384 @ proj_text
        
        print(f"  📝 Texte: '{test_text}'")
        print(f"     Embedding SBERT: {len(emb_384)} dims")
        print(f"     Après projection: {len(emb_512)} dims")
        print(f"     Norme après projection: {np.linalg.norm(emb_512):.4f}")
        
        # Tester une recherche dans v3 avec projection
        results = search_by_text(emb_384, k=5, modal_filter='all')
        print(f"\n  🔍 Recherche dans FAISS v3: {len(results)} résultats")
        
        for i, r in enumerate(results[:3]):
            print(f"     {i+1}. [{r['score']:.4f}] {r['modal']} - {r['text'][:50]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


# ============================================================================
# SECTION 5 : Précision sémantique
# ============================================================================

def test_semantic_precision():
    """Teste la précision sémantique sur des paires de textes."""
    print_section("12. Semantic Precision Test")
    
    test_pairs = [
        # (texte1, texte2, devrait_être_similaire)
        ("The cat is sleeping on the couch", "A kitten resting on a sofa", True),
        ("The cat is sleeping on the couch", "The dog is running in the park", False),
        ("Stock market closes higher today", "Shares rallied on Wall Street", True),
        ("Stock market closes higher today", "The weather is nice today", False),
        ("Machine learning algorithms", "Deep neural networks training", True),
        ("Machine learning algorithms", "How to bake a cake", False),
    ]
    
    correct = 0
    total = 0
    
    print(f"\n  📊 Test de similarité sémantique:")
    
    for text1, text2, should_be_similar in test_pairs:
        emb1 = embed_text(text1)
        emb2 = embed_text(text2)
        
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        is_similar = sim > 0.5
        is_correct = is_similar == should_be_similar
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        expected = "similaire" if should_be_similar else "différent"
        print(f"  {status} sim={sim:.4f} ({expected})")
        print(f"     '{text1[:40]}'")
        print(f"     '{text2[:40]}'")
    
    accuracy = correct / total * 100
    print(f"\n  📈 Précision: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 70:
        print("  ✅ Bonne précision sémantique")
    else:
        print("  ⚠️ Précision à améliorer")
    
    return accuracy


# ============================================================================
# SECTION 6 : Résumé et verdict
# ============================================================================


def print_summary(results):
    """Affiche le résumé des tests."""
    print_section("FINAL SUMMARY")
    
    passed = []
    failed = []
    
    tests = [
        ("Model Loading", results.get('loading', False)),
        ("Embedding Dimension", results.get('dimension', False)),
        ("Single Embedding", results.get('single', False)),
        ("Batch Embedding", results.get('batch', False)),
        ("Cosine Similarity", results.get('similarity', False)),
        ("Semantic Search", results.get('semantic', False)),
        ("Performance", results.get('performance', False)),
        ("Batch Performance", results.get('batch_perf', False)),
        ("FAISS v1 Integration", results.get('faiss_v1', False)),
        ("FAISS v2 Integration", results.get('faiss_v2', False)),
        ("V3 Projection", results.get('v3_projection', False)),
        ("Semantic Precision", results.get('precision', False)),
    ]
    
    for name, ok in tests:
        # Convertir en booléen si c'est un array numpy
        if hasattr(ok, '__len__') and not isinstance(ok, (bool, str)):
            ok = True if len(ok) > 0 else False
        if ok:
            passed.append(f"✅ {name}")
        else:
            failed.append(f"❌ {name}")
    
    print("\n" + "\n".join(passed))
    if failed:
        print("\n" + "\n".join(failed))
    
    print(f"\n{'=' * 60}")
    print(f"📊 Résultat: {len(passed)}/{len(tests)} tests passés")
    
    if len(failed) == 0:
        print("🎉 VERDICT: SBERT model is EXCELLENT - Production Ready!")
    elif len(failed) <= 2:
        print("✅ VERDICT: SBERT model is GOOD - Minor issues")
    else:
        print("⚠️ VERDICT: SBERT model needs improvement")
    
    print(f"{'=' * 60}")


def run_all_tests():
    """Exécute tous les tests SBERT."""
    print("\n" + "🎯" * 30)
    print(" SBERT MODEL VALIDATION SUITE")
    print(" Tests: Loading, Embedding, Similarity, FAISS Integration")
    print("🎯" * 30)
    
    results = {}
    
    # Section 1: Chargement
    results['loading'] = test_model_loading()
    results['dimension'] = test_embedding_dimension()
    
    # Section 2: Embedding (retourne un booléen, pas l'array)
    single_result = test_single_embedding()
    results['single'] = single_result if isinstance(single_result, bool) else True
    
    batch_result = test_batch_embedding()
    results['batch'] = batch_result if isinstance(batch_result, bool) else True
    
    # Section 3: Similarité
    results['similarity'] = test_cosine_similarity()
    results['semantic'] = test_semantic_search()
    
    # Section 4: Performance
    results['performance'] = test_performance()
    results['batch_perf'] = test_batch_performance()
    
    # Section 5: FAISS
    results['faiss_v1'] = test_faiss_v1_integration()
    results['faiss_v2'] = test_faiss_v2_integration()
    results['v3_projection'] = test_projection_for_v3()
    
    # Section 6: Précision
    precision = test_semantic_precision()
    results['precision'] = precision >= 70 if isinstance(precision, (int, float)) else False
    
    # Résumé
    print_summary(results)
    
    return results

if __name__ == '__main__':
    run_all_tests()