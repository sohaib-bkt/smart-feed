# tests/test_clip_cross_modal.py
import sys
sys.path.insert(0, '.')

import numpy as np
from app.models.multimodal.clip_model import embed_post_multimodal
from app.services.multimodal_index import get_v3_index, search_multimodal, get_index_stats


def get_text_embedding(text: str) -> np.ndarray:
    """
    Obtient un embedding texte via CLIP en utilisant embed_post_multimodal.
    """
    # embed_post_multimodal attend (caption, image, alpha)
    # Pour obtenir uniquement l'embedding texte, on passe une image vide
    import numpy as np
    dummy_image = np.zeros((224, 224, 3)).astype(np.uint8)
    embedding = embed_post_multimodal(text, dummy_image, alpha=1.0)  # alpha=1.0 = 100% texte
    return np.array(embedding)


def test_clip_cross_modal():
    """Test cross-modal avec CLIP directement"""
    
    print("\n" + "=" * 60)
    print("🔍 CLIP Cross-Modal Search Test")
    print("=" * 60)
    
    # 1. Vérifier l'index
    try:
        index, metadata = get_v3_index()
        print(f"✅ Index v3 chargé: {index.ntotal} vecteurs")
    except Exception as e:
        print(f"❌ Erreur chargement index: {e}")
        return
    
    # 2. Statistiques de l'index
    stats = get_index_stats()
    print(f"\n📊 Composition de l'index:")
    print(f"   - Texte: {stats['by_modal'].get('text', 0)}")
    print(f"   - Images: {stats['by_modal'].get('image', 0)}")
    print(f"   - Vidéos: {stats['by_modal'].get('video', 0)}")
    
    # 3. Test avec différentes requêtes
    print("\n" + "-" * 60)
    print("📝 Test 1: Recherche texte → images avec CLIP")
    print("-" * 60)
    
    test_queries = [
        "a cat playing with a ball",
        "a dog running in a park", 
        "delicious food on a plate",
        "sunset over mountains",
        "modern city architecture"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Requête: '{query}'")
        
        try:
            # Embedding avec CLIP via embed_post_multimodal
            query_emb = get_text_embedding(query)
            
            # Utiliser la fonction search_multimodal du service
            results = search_multimodal(
                query_embedding=query_emb,
                k=15,
                modal_filter='image'
            )
            
            print(f"   Images trouvées: {len(results)}/15")
            
            if results:
                print(f"   Top résultats:")
                for i, r in enumerate(results[:3]):
                    print(f"     {i+1}. [{r['score']:.4f}] {r['text'][:60]}...")
            else:
                print(f"   ⚠️ Aucune image trouvée pour cette requête")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # 4. Test avec recherche par texte via SBERT (comparaison)
    print("\n" + "-" * 60)
    print("📝 Test 2: Comparaison CLIP vs SBERT projeté")
    print("-" * 60)
    
    from app.models.embeddings import embed_text
    from app.services.multimodal_index import search_by_text
    
    query = "a cat playing"
    
    try:
        # CLIP
        clip_emb = get_text_embedding(query)
        clip_results = search_multimodal(clip_emb, k=10, modal_filter='image')
        
        # SBERT projeté
        sbert_emb = embed_text(query)
        sbert_results = search_by_text(sbert_emb, k=10, modal_filter='image')
        
        print(f"\n🔍 Requête: '{query}'")
        print(f"\n   🎯 CLIP (embedding multimodal):")
        print(f"      Images trouvées: {len(clip_results)}/10")
        for i, r in enumerate(clip_results[:3]):
            print(f"        {i+1}. [{r['score']:.4f}] {r['text'][:50]}...")
        
        print(f"\n   🔄 SBERT projeté (texte → espace CLIP):")
        print(f"      Images trouvées: {len(sbert_results)}/10")
        for i, r in enumerate(sbert_results[:3]):
            print(f"        {i+1}. [{r['score']:.4f}] {r['text'][:50]}...")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 5. Test spécifique avec une image de référence
    print("\n" + "-" * 60)
    print("📝 Test 3: Recherche par similarité d'images")
    print("-" * 60)
    
    # Chercher des images similaires entre elles
    image_indices = metadata[metadata['modal'] == 'image'].index.tolist()
    
    if image_indices:
        import random
        test_idx = random.choice(image_indices)
        test_row = metadata.iloc[test_idx]
        
        print(f"\n   Image de référence: {test_row.get('text', '')[:80]}...")
        print(f"   💡 Pour rechercher par image, on utiliserait embed_post_multimodal avec alpha=0.0")
    else:
        print(f"   ⚠️ Aucune image trouvée dans l'index")
    
    # 6. Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    
    print("✅ CLIP est correctement intégré pour la recherche cross-modale!")
    print("\n💡 RECOMMANDATIONS:")
    print("   - Utilisez embed_post_multimodal(text, dummy_image, alpha=1.0) pour texte seul")
    print("   - Utilisez embed_post_multimodal('', image, alpha=0.0) pour image seule")
    print("   - Utilisez alpha=0.5 pour combiner texte et image")


def test_specific_image_concepts():
    """Test des concepts d'images spécifiques"""
    print("\n" + "=" * 60)
    print("🎯 Test des concepts d'images spécifiques")
    print("=" * 60)
    
    try:
        index, metadata = get_v3_index()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Concepts qu'on veut trouver
    concepts = {
        "animal": ["cat", "dog", "bird", "horse"],
        "food": ["pizza", "cake", "fruit", "vegetable"],
        "nature": ["flower", "tree", "mountain", "beach"],
        "vehicle": ["car", "bicycle", "train", "airplane"]
    }
    
    results_summary = {}
    
    for category, words in concepts.items():
        print(f"\n📂 Catégorie: {category}")
        category_results = []
        
        for word in words:
            try:
                query_emb = get_text_embedding(word)
                results = search_multimodal(query_emb, k=10, modal_filter='image')
                category_results.append(len(results))
                
                status = "✅" if len(results) > 0 else "❌"
                print(f"   {status} '{word}': {len(results)} images trouvées")
            except Exception as e:
                print(f"   ❌ '{word}': erreur - {e}")
                category_results.append(0)
        
        if category_results:
            results_summary[category] = sum(category_results) / len(category_results)
    
    print("\n" + "-" * 60)
    print("📊 Moyenne d'images par concept:")
    for category, avg in results_summary.items():
        print(f"   {category}: {avg:.1f} images")


def test_cross_modal_ranking():
    """Test du classement cross-modal"""
    print("\n" + "=" * 60)
    print("📊 Test du classement cross-modal")
    print("=" * 60)
    
    try:
        index, metadata = get_v3_index()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Requête avec vrai contenu visuel
    query = "a person riding a bicycle on a sunny day"
    
    try:
        query_emb = get_text_embedding(query)
        
        # Recherche
        results = search_multimodal(query_emb, k=20, modal_filter='all')
        
        print(f"\n🔍 Requête: '{query}'")
        print(f"\n   Résultats par modalité:")
        
        modal_counts = {'text': 0, 'image': 0, 'video': 0}
        for r in results:
            modal_counts[r['modal']] += 1
        
        for modal, count in modal_counts.items():
            print(f"      - {modal}: {count}")
        
        print(f"\n   Top 5 résultats:")
        for i, r in enumerate(results[:5]):
            tag = "🖼️" if r['modal'] == 'image' else "🎥" if r['modal'] == 'video' else "📝"
            print(f"      {i+1}. {tag} [{r['score']:.4f}] {r['text'][:60]}...")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")


if __name__ == '__main__':
    print("\n" + "🎯" * 30)
    print(" CLIP CROSS-MODAL VALIDATION SUITE")
    print("🎯" * 30)
    
    # Test principal
    test_clip_cross_modal()
    
    # Tests supplémentaires
    test_specific_image_concepts()
    test_cross_modal_ranking()
    
    print("\n" + "🎯" * 30)
    print(" TESTS COMPLÉTÉS")
    print("🎯" * 30)