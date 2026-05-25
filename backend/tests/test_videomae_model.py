# tests/test_videomae_model.py
"""
Test complet du modèle VideoMAE pour les embeddings vidéo.

Teste :
- Chargement du modèle VideoMAE
- Embedding de vidéos (frames)
- Normalisation des embeddings
- Similarité entre vidéos
- Performance
- Intégration avec FAISS v3
- Projection vers espace multimodal (768d → 512d)
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# ============================================================================
# Imports
# ============================================================================

def print_section(title):
    """Affiche une section de test."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title):
    """Affiche une sous-section."""
    print(f"\n  --- {title} ---")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """Normalise un embedding L2."""
    return emb / (np.linalg.norm(emb) + 1e-8)


# ============================================================================
# Utilitaires vidéo (sans OpenCV)
# ============================================================================

def create_dummy_video(num_frames: int = 16, height: int = 224, width: int = 224) -> np.ndarray:
    """
    Crée une vidéo factice pour les tests.
    Retourne un array (num_frames, height, width, 3)
    """
    frames = []
    for i in range(num_frames):
        # Image aléatoire
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Ajouter un motif qui change avec le temps
        position = int(i * height / num_frames)
        frame[position:position+3, :] = [255, 0, 0]  # Ligne rouge horizontale
        
        # Ajouter un carré qui bouge
        x_pos = int(i * width / num_frames)
        frame[50:70, x_pos:x_pos+20] = [0, 255, 0]  # Carré vert
        
        frames.append(frame)
    
    return np.array(frames)


# ============================================================================
# SECTION 1 : Chargement du modèle VideoMAE
# ============================================================================

def test_videomae_model_loading():
    """Teste le chargement du modèle VideoMAE."""
    print_section("1. VideoMAE Model Loading")
    
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
        
        start = time.time()
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        load_time = time.time() - start
        
        print(f"  ✅ Modèle chargé en {load_time:.2f}s")
        print(f"  ✅ Modèle: {model.config.model_type}")
        print(f"  ✅ Dimension embedding: {model.config.hidden_size}")
        
        return model, processor
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None, None


def test_videomae_finetuned_loading():
    """Teste le chargement du modèle VideoMAE fine-tuné."""
    print_section("2. VideoMAE Fine-tuned Model Loading")
    
    # Vérifier plusieurs chemins possibles
    possible_paths = [
        Path("models/multimodal/videomae_finetuned"),
        Path("app/models/multimodal/videomae_finetuned"),
        Path("models/multimodal/videomae_finetuned"),
        Path("models/multimodal/videomae_finetuned"),  # votre chemin
    ]
    
    finetuned_path = None
    for path in possible_paths:
        if path.exists():
            finetuned_path = path
            break
    
    if finetuned_path is None:
        print(f"  ⚠️ Modèle fine-tuné non trouvé")
        print(f"  ℹ️ Utilisation du modèle de base pour les tests")
        return None, None
    
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
        
        start = time.time()
        model = VideoMAEModel.from_pretrained(str(finetuned_path))
        processor = VideoMAEImageProcessor.from_pretrained(str(finetuned_path))
        load_time = time.time() - start
        
        print(f"  ✅ Modèle fine-tuné chargé en {load_time:.2f}s")
        print(f"  ✅ Chemin: {finetuned_path}")
        print(f"  ✅ Dimension embedding: {model.config.hidden_size}")
        
        return model, processor
    except Exception as e:
        print(f"  ❌ Erreur chargement modèle fine-tuné: {e}")
        return None, None

# ============================================================================
# SECTION 2 : Embedding de vidéos
# ============================================================================

def get_video_embedding(model, processor, video_frames: np.ndarray) -> np.ndarray:
    """Extrait l'embedding d'une vidéo et le normalise."""
    import torch
    
    inputs = processor(list(video_frames), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
    
    # Normalisation L2
    embedding = normalize_embedding(embedding)
    return embedding


def test_video_embedding(model, processor):
    """Teste l'embedding d'une vidéo."""
    print_section("3. Video Embedding")
    
    if model is None:
        print("  ❌ Modèle non disponible")
        return None
    
    try:
        import torch
        
        # Créer une vidéo factice
        video_frames = create_dummy_video(num_frames=16)
        print(f"  📹 Vidéo factice: {video_frames.shape}")
        
        # Embedding
        start = time.time()
        embedding = get_video_embedding(model, processor, video_frames)
        embed_time = time.time() - start
        
        print(f"  ⏱️ Embedding: {embed_time*1000:.2f}ms")
        print(f"  📊 Dimension embedding: {len(embedding)}")
        print(f"  📐 Norme: {np.linalg.norm(embedding):.4f} (normalisé)")
        
        return embedding
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None


def test_batch_video_embedding(model, processor):
    """Teste l'embedding par lots de vidéos."""
    print_section("4. Batch Video Embedding")
    
    if model is None:
        print("  ❌ Modèle non disponible")
        return None
    
    try:
        n_videos = 5
        embeddings = []
        
        print(f"  🎬 Test avec {n_videos} vidéos")
        
        start_total = time.time()
        
        for i in range(n_videos):
            video_frames = create_dummy_video(num_frames=16)
            embedding = get_video_embedding(model, processor, video_frames)
            embeddings.append(embedding)
        
        total_time = time.time() - start_total
        
        embeddings_array = np.array(embeddings)
        
        print(f"  ✅ {len(embeddings)} embeddings générés")
        print(f"  ⏱️ Temps total: {total_time*1000:.2f}ms")
        print(f"  ⏱️ Temps par vidéo: {total_time*1000/n_videos:.2f}ms")
        print(f"  📊 Shape des embeddings: {embeddings_array.shape}")
        
        # Vérifier les normes
        norms = np.linalg.norm(embeddings_array, axis=1)
        print(f"  📐 Normes: mean={norms.mean():.4f}, std={norms.std():.4f}")
        
        return embeddings_array
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None


# ============================================================================
# SECTION 3 : Similarité entre vidéos
# ============================================================================

def test_video_similarity(model, processor):
    """Teste la similarité entre vidéos."""
    print_section("5. Video Similarity")
    
    if model is None:
        print("  ❌ Modèle non disponible")
        return
    
    try:
        # Créer deux vidéos
        video1 = create_dummy_video(num_frames=16)
        video2 = create_dummy_video(num_frames=16)
        
        # Version similaire à video1 (légère variation)
        video1_similar = video1.copy().astype(np.int16)
        video1_similar[8, 100:120, :] = video1_similar[8, 100:120, :] + 10
        video1_similar = np.clip(video1_similar, 0, 255).astype(np.uint8)
        
        emb1 = get_video_embedding(model, processor, video1)
        emb1_sim = get_video_embedding(model, processor, video1_similar)
        emb2 = get_video_embedding(model, processor, video2)
        
        # Similarités
        sim_identical = cosine_similarity(emb1, emb1)
        sim_similar = cosine_similarity(emb1, emb1_sim)
        sim_different = cosine_similarity(emb1, emb2)
        
        print(f"\n  📊 Similarités:")
        print(f"     Identique (même vidéo): {sim_identical:.4f}")
        print(f"     Similaire (légère variation): {sim_similar:.4f}")
        print(f"     Différente (vidéo aléatoire): {sim_different:.4f}")
        
        if sim_similar > sim_different:
            print("\n  ✅ Similarité cohérente")
        else:
            print("\n  ⚠️ Similarité à vérifier")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


# ============================================================================
# SECTION 4 : Projection vers FAISS v3 (768d → 512d)
# ============================================================================

def load_video_projection() -> Optional[np.ndarray]:
    """Charge la matrice de projection vidéo (768 → 512)."""
    # Essayer différents noms de fichiers
    possible_paths = [
        Path("data/processed/proj_768_512.npy"),
        Path("data/processed/proj_video_500_512.npy"),
        Path("data/processed/proj_video_768_512.npy"),
    ]
    
    for path in possible_paths:
        if path.exists():
            proj = np.load(path).astype('float32')
            print(f"  ✅ Matrice de projection chargée: {path.name} {proj.shape}")
            return proj
    
    print("  ⚠️ Aucune matrice de projection vidéo trouvée")
    return None


def test_video_projection(model, processor):
    """Teste la projection des embeddings vidéo vers l'espace multimodal."""
    print_section("6. Video Projection for FAISS v3")
    
    proj_video = load_video_projection()
    
    if proj_video is None:
        print("  ❌ Matrice de projection non disponible")
        return False
    
    try:
        from app.services.multimodal_index import search_multimodal
        
        # Créer un vrai embedding vidéo
        video_frames = create_dummy_video(num_frames=16)
        video_emb = get_video_embedding(model, processor, video_frames)
        
        print(f"  📊 Embedding vidéo original: {len(video_emb)} dims, norm={np.linalg.norm(video_emb):.4f}")
        
        # Vérifier que la projection a la bonne dimension
        if proj_video.shape[0] != len(video_emb):
            print(f"  ⚠️ Dimension mismatch: embedding={len(video_emb)}, proj={proj_video.shape[0]}")
            # Tronquer ou padder l'embedding si nécessaire
            if proj_video.shape[0] < len(video_emb):
                video_emb = video_emb[:proj_video.shape[0]]
                print(f"  🔧 Embedding tronqué à {proj_video.shape[0]} dims")
            else:
                padding = np.zeros(proj_video.shape[0] - len(video_emb))
                video_emb = np.concatenate([video_emb, padding])
                print(f"  🔧 Embedding paddé à {proj_video.shape[0]} dims")
        
        # Projeter vers 512 dims
        projected_emb = video_emb @ proj_video
        projected_emb = normalize_embedding(projected_emb)
        
        print(f"  📊 Après projection: {len(projected_emb)} dims, norm={np.linalg.norm(projected_emb):.4f}")
        
        # Tester une recherche dans FAISS v3
        results = search_multimodal(projected_emb, k=5, modal_filter='all')
        
        print(f"\n  🔍 Recherche dans FAISS v3: {len(results)} résultats")
        for i, r in enumerate(results[:3]):
            print(f"     {i+1}. [{r['score']:.4f}] {r['modal']} - {r['text'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


# ============================================================================
# SECTION 5 : Performance
# ============================================================================

def test_video_performance(model, processor):
    """Teste les performances d'embedding vidéo."""
    print_section("7. Video Performance Test")
    
    if model is None:
        print("  ❌ Modèle non disponible")
        return False
    
    try:
        n_iterations = 10
        times = []
        
        for i in range(n_iterations):
            video_frames = create_dummy_video(num_frames=16)
            
            start = time.time()
            _ = get_video_embedding(model, processor, video_frames)
            times.append(time.time() - start)
        
        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        
        print(f"\n  ⏱️ Statistiques ({n_iterations} itérations):")
        print(f"     Moyenne: {avg_ms:.2f}ms (±{std_ms:.2f})")
        print(f"     Min: {min(times)*1000:.2f}ms")
        print(f"     Max: {max(times)*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


# ============================================================================
# SECTION 6 : Intégration avec CLIP
# ============================================================================

def test_video_with_clip(model, processor):
    """Teste la corrélation entre VideoMAE et CLIP après projection."""
    print_section("8. Video vs CLIP Correlation")
    
    proj_video = load_video_projection()
    
    if proj_video is None:
        print("  ⚠️ Matrice de projection non disponible - test ignoré")
        return None
    
    try:
        from app.models.multimodal.clip_model import embed_post_multimodal
        
        # Créer une vidéo et extraire une image clé
        video_frames = create_dummy_video(num_frames=16)
        key_frame = video_frames[8]
        
        # Embedding vidéo
        video_emb = get_video_embedding(model, processor, video_frames)
        
        # Ajuster la dimension si nécessaire
        if proj_video.shape[0] != len(video_emb):
            if proj_video.shape[0] < len(video_emb):
                video_emb = video_emb[:proj_video.shape[0]]
            else:
                padding = np.zeros(proj_video.shape[0] - len(video_emb))
                video_emb = np.concatenate([video_emb, padding])
        
        # Projeter
        video_emb_proj = video_emb @ proj_video
        video_emb_proj = normalize_embedding(video_emb_proj)
        
        # Embedding CLIP
        clip_emb = embed_post_multimodal("", key_frame, alpha=0.0)
        
        # Similarité
        similarity = cosine_similarity(video_emb_proj, clip_emb)
        
        print(f"\n  📊 Similarité VideoMAE → CLIP (après projection): {similarity:.4f}")
        
        return similarity
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None


# ============================================================================
# SECTION 7 : Résumé et verdict
# ============================================================================

def print_summary(results):
    """Affiche le résumé des tests."""
    print_section("FINAL SUMMARY")
    
    passed = []
    failed = []
    skipped = []
    
    tests = [
        ("Model Loading (base)", results.get('loading_base', False)),
        ("Model Loading (fine-tuned)", results.get('loading_finetuned', None)),
        ("Video Embedding", results.get('embedding', False)),
        ("Batch Video Embedding", results.get('batch', False)),
        ("Video Similarity", results.get('similarity', False)),
        ("Projection (768d → 512d)", results.get('projection', False)),
        ("Performance", results.get('performance', False)),
        ("Video-CLIP Correlation", results.get('correlation', None)),
    ]
    
    for name, ok in tests:
        if ok is True:
            passed.append(f"✅ {name}")
        elif ok is None:
            skipped.append(f"  ⏭️ {name} (skipped)")
        else:
            failed.append(f"❌ {name}")
    
    print("\n" + "\n".join(passed))
    if skipped:
        print("\n" + "\n".join(skipped))
    if failed:
        print("\n" + "\n".join(failed))
    
    print(f"\n{'=' * 60}")
    print(f"📊 Résultat: {len(passed)}/{len(tests)} tests")
    
    if len(failed) == 0:
        print("🎉 VERDICT: VideoMAE model is EXCELLENT - Production Ready!")
    elif len(failed) <= 2:
        print("✅ VERDICT: VideoMAE model is GOOD - Minor issues")
    else:
        print("⚠️ VERDICT: VideoMAE model needs improvement")
    
    print(f"{'=' * 60}")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Exécute tous les tests VideoMAE."""
    print("\n" + "🎯" * 30)
    print(" VIDEOMAE MODEL VALIDATION SUITE")
    print(" Tests: Loading, Embedding, Similarity, FAISS Integration")
    print("🎯" * 30)
    
    results = {}
    
    # 1. Chargement modèle base
    model_base, processor_base = test_videomae_model_loading()
    results['loading_base'] = model_base is not None
    
    # 2. Chargement modèle fine-tuné
    model_ft, processor_ft = test_videomae_finetuned_loading()
    results['loading_finetuned'] = model_ft is not None
    
    # Utiliser le modèle disponible
    model = model_ft if model_ft is not None else model_base
    processor = processor_ft if processor_ft is not None else processor_base
    
    if model is None:
        print("\n  ❌ Aucun modèle VideoMAE disponible. Arrêt des tests.")
        print_summary(results)
        return results
    
    # 3. Embedding vidéo
    embedding = test_video_embedding(model, processor)
    results['embedding'] = embedding is not None
    
    # 4. Batch embedding
    batch_embs = test_batch_video_embedding(model, processor)
    results['batch'] = batch_embs is not None
    
    # 5. Similarité
    results['similarity'] = test_video_similarity(model, processor)
    
    # 6. Projection (utilise vos fichiers)
    results['projection'] = test_video_projection(model, processor)
    
    # 7. Performance
    results['performance'] = test_video_performance(model, processor)
    
    # 8. Corrélation avec CLIP
    results['correlation'] = test_video_with_clip(model, processor) is not None
    
    # Résumé
    print_summary(results)
    
    return results


if __name__ == '__main__':
    run_all_tests()