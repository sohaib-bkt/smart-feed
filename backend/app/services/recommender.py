import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from app.services.scoring import compute_score
from app.services.explainability import annotate_feed
from app.config import settings

# Charger index et métadonnées au démarrage (singleton)
_index = None
_posts_df = None
_index_v2 = None
_posts_df_v2 = None

# ── Media URL resolvers ──────────────────────────────────────────────────────
# Public sample videos (CDNs verified to work cross-origin)
UCF101_VIDEO_URLS: dict[str, str] = {
    "Diving":        "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "PlayingGuitar": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "TennisSwing":   "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "Basketball":    "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "YoYo":          "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "Skijet":        "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "HorseRiding":   "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "WalkingWithDog":"https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
}

# Pexels thumbnail images (these do work cross-origin via CDN)
UCF101_THUMB_URLS: dict[str, str] = {
    "Diving":        "https://images.pexels.com/photos/261185/pexels-photo-261185.jpeg?auto=compress&w=800",
    "PlayingGuitar": "https://images.pexels.com/photos/210922/pexels-photo-210922.jpeg?auto=compress&w=800",
    "TennisSwing":   "https://images.pexels.com/photos/209977/pexels-photo-209977.jpeg?auto=compress&w=800",
    "Basketball":    "https://images.pexels.com/photos/358042/pexels-photo-358042.jpeg?auto=compress&w=800",
    "YoYo":          "https://images.pexels.com/photos/3661193/pexels-photo-3661193.jpeg?auto=compress&w=800",
    "Skijet":        "https://images.pexels.com/photos/1430672/pexels-photo-1430672.jpeg?auto=compress&w=800",
    "HorseRiding":   "https://images.pexels.com/photos/1996332/pexels-photo-1996332.jpeg?auto=compress&w=800",
    "WalkingWithDog":"https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&w=800",
}

# COCO captions → fallback category Pexels image when text matching fails
CATEGORY_IMAGE_URLS: dict[str, str] = {
    "POLITICS":         "https://images.pexels.com/photos/261949/pexels-photo-261949.jpeg?auto=compress&w=800",
    "WELLNESS":         "https://images.pexels.com/photos/3829227/pexels-photo-3829227.jpeg?auto=compress&w=800",
    "ENTERTAINMENT":    "https://images.pexels.com/photos/1763075/pexels-photo-1763075.jpeg?auto=compress&w=800",
    "TRAVEL":           "https://images.pexels.com/photos/739407/pexels-photo-739407.jpeg?auto=compress&w=800",
    "STYLE & BEAUTY":   "https://images.pexels.com/photos/1055691/pexels-photo-1055691.jpeg?auto=compress&w=800",
    "PARENTING":        "https://images.pexels.com/photos/3933250/pexels-photo-3933250.jpeg?auto=compress&w=800",
    "HEALTHY LIVING":   "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&w=800",
    "QUEER VOICES":     "https://images.pexels.com/photos/1153369/pexels-photo-1153369.jpeg?auto=compress&w=800",
    "FOOD & DRINK":     "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&w=800",
    "BUSINESS":         "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg?auto=compress&w=800",
    "COMEDY":           "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg?auto=compress&w=800",
    "SPORTS":           "https://images.pexels.com/photos/4775193/pexels-photo-4775193.jpeg?auto=compress&w=800",
    "BLACK VOICES":     "https://images.pexels.com/photos/1153369/pexels-photo-1153369.jpeg?auto=compress&w=800",
    "HOME & LIVING":    "https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&w=800",
    "PARENTS":          "https://images.pexels.com/photos/3933250/pexels-photo-3933250.jpeg?auto=compress&w=800",
    "TIPS":             "https://images.pexels.com/photos/414423/pexels-photo-414423.jpeg?auto=compress&w=800",
    "WEDDINGS":         "https://images.pexels.com/photos/225384/pexels-photo-225384.jpeg?auto=compress&w=800",
    "IMPACT":           "https://images.pexels.com/photos/318236/pexels-photo-318236.jpeg?auto=compress&w=800",
    "DIVORCE":          "https://images.pexels.com/photos/3771085/pexels-photo-3771085.jpeg?auto=compress&w=800",
    "CRIME":            "https://images.pexels.com/photos/87651/earth-blue-planet-globe-planet-87651.jpeg?auto=compress&w=800",
    "MEDIA":            "https://images.pexels.com/photos/3568520/pexels-photo-3568520.jpeg?auto=compress&w=800",
    "WEIRD NEWS":       "https://images.pexels.com/photos/2526935/pexels-photo-2526935.jpeg?auto=compress&w=800",
    "GREEN":            "https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg?auto=compress&w=800",
    "WORLDPOST":        "https://images.pexels.com/photos/87651/earth-blue-planet-globe-planet-87651.jpeg?auto=compress&w=800",
    "THE WORLDPOST":    "https://images.pexels.com/photos/87651/earth-blue-planet-globe-planet-87651.jpeg?auto=compress&w=800",
    "RELIGION":         "https://images.pexels.com/photos/2227778/pexels-photo-2227778.jpeg?auto=compress&w=800",
    "STYLE":            "https://images.pexels.com/photos/1055691/pexels-photo-1055691.jpeg?auto=compress&w=800",
    "SCIENCE":          "https://images.pexels.com/photos/2280549/pexels-photo-2280549.jpeg?auto=compress&w=800",
    "WORLD NEWS":       "https://images.pexels.com/photos/87651/earth-blue-planet-globe-planet-87651.jpeg?auto=compress&w=800",
    "TASTE":            "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&w=800",
    "TECH":             "https://images.pexels.com/photos/3568518/pexels-photo-3568518.jpeg?auto=compress&w=800",
    "MONEY":            "https://images.pexels.com/photos/259200/pexels-photo-259200.jpeg?auto=compress&w=800",
    "ARTS":             "https://images.pexels.com/photos/1509428/pexels-photo-1509428.jpeg?auto=compress&w=800",
    "FIFTY":            "https://images.pexels.com/photos/3585047/pexels-photo-3585047.jpeg?auto=compress&w=800",
    "GOOD NEWS":        "https://images.pexels.com/photos/267389/pexels-photo-267389.jpeg?auto=compress&w=800",
    "ARTS & CULTURE":   "https://images.pexels.com/photos/1509428/pexels-photo-1509428.jpeg?auto=compress&w=800",
    "ENVIRONMENT":      "https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg?auto=compress&w=800",
    "COLLEGE":          "https://images.pexels.com/photos/267885/pexels-photo-267885.jpeg?auto=compress&w=800",
    "LATINO VOICES":    "https://images.pexels.com/photos/318236/pexels-photo-318236.jpeg?auto=compress&w=800",
    "CULTURE & ARTS":   "https://images.pexels.com/photos/1509428/pexels-photo-1509428.jpeg?auto=compress&w=800",
    "EDUCATION":        "https://images.pexels.com/photos/267885/pexels-photo-267885.jpeg?auto=compress&w=800",
    "WOMEN":            "https://images.pexels.com/photos/3585047/pexels-photo-3585047.jpeg?auto=compress&w=800",
    "tifu":             "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg?auto=compress&w=800",
    "visual":           "https://images.pexels.com/photos/3568520/pexels-photo-3568520.jpeg?auto=compress&w=800",
}

# COCO caption keyword → Pexels image fallback (since COCO images have no direct CDN URL we want to ship)
COCO_KEYWORD_IMAGES: dict[str, str] = {
    "motorcycle":  "https://images.pexels.com/photos/1715193/pexels-photo-1715193.jpeg?auto=compress&w=800",
    "cubicle":     "https://images.pexels.com/photos/3568520/pexels-photo-3568520.jpeg?auto=compress&w=800",
    "toilet":      "https://images.pexels.com/photos/6585629/pexels-photo-6585629.jpeg?auto=compress&w=800",
    "bench":       "https://images.pexels.com/photos/2581922/pexels-photo-2581922.jpeg?auto=compress&w=800",
    "dessert":     "https://images.pexels.com/photos/291528/pexels-photo-291528.jpeg?auto=compress&w=800",
    "kitchen":     "https://images.pexels.com/photos/1909791/pexels-photo-1909791.jpeg?auto=compress&w=800",
    "car":         "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg?auto=compress&w=800",
    "dog":         "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&w=800",
    "cat":         "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&w=800",
    "beach":       "https://images.pexels.com/photos/1450082/pexels-photo-1450082.jpeg?auto=compress&w=800",
    "mountain":    "https://images.pexels.com/photos/1366919/pexels-photo-1366919.jpeg?auto=compress&w=800",
    "city":        "https://images.pexels.com/photos/466685/pexels-photo-466685.jpeg?auto=compress&w=800",
    "building":    "https://images.pexels.com/photos/466685/pexels-photo-466685.jpeg?auto=compress&w=800",
    "tree":        "https://images.pexels.com/photos/38136/pexels-photo-38136.jpeg?auto=compress&w=800",
    "flower":      "https://images.pexels.com/photos/56866/garden-rose-red-pink-56866.jpeg?auto=compress&w=800",
    "pizza":       "https://images.pexels.com/photos/315755/pexels-photo-315755.jpeg?auto=compress&w=800",
    "food":        "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&w=800",
    "phone":       "https://images.pexels.com/photos/607812/pexels-photo-607812.jpeg?auto=compress&w=800",
    "laptop":      "https://images.pexels.com/photos/18105/pexels-photo.jpg?auto=compress&w=800",
    "computer":    "https://images.pexels.com/photos/3568518/pexels-photo-3568518.jpeg?auto=compress&w=800",
    "book":        "https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg?auto=compress&w=800",
    "people":      "https://images.pexels.com/photos/318236/pexels-photo-318236.jpeg?auto=compress&w=800",
    "person":      "https://images.pexels.com/photos/318236/pexels-photo-318236.jpeg?auto=compress&w=800",
    "man":         "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&w=800",
    "woman":       "https://images.pexels.com/photos/3585047/pexels-photo-3585047.jpeg?auto=compress&w=800",
    "child":       "https://images.pexels.com/photos/3933250/pexels-photo-3933250.jpeg?auto=compress&w=800",
    "baby":        "https://images.pexels.com/photos/3933250/pexels-photo-3933250.jpeg?auto=compress&w=800",
    "sport":       "https://images.pexels.com/photos/4775193/pexels-photo-4775193.jpeg?auto=compress&w=800",
    "soccer":      "https://images.pexels.com/photos/4775193/pexels-photo-4775193.jpeg?auto=compress&w=800",
    "baseball":    "https://images.pexels.com/photos/4775193/pexels-photo-4775193.jpeg?auto=compress&w=800",
    "tennis":      "https://images.pexels.com/photos/209977/pexels-photo-209977.jpeg?auto=compress&w=800",
    "gym":         "https://images.pexels.com/photos/791763/pexels-photo-791763.jpeg?auto=compress&w=800",
    "music":       "https://images.pexels.com/photos/210922/pexels-photo-210922.jpeg?auto=compress&w=800",
    "guitar":      "https://images.pexels.com/photos/210922/pexels-photo-210922.jpeg?auto=compress&w=800",
    "piano":       "https://images.pexels.com/photos/210922/pexels-photo-210922.jpeg?auto=compress&w=800",
    "bird":        "https://images.pexels.com/photos/326900/pexels-photo-326900.jpeg?auto=compress&w=800",
    "skate":       "https://images.pexels.com/photos/4775193/pexels-photo-4775193.jpeg?auto=compress&w=800",
    "surf":        "https://images.pexels.com/photos/1450082/pexels-photo-1450082.jpeg?auto=compress&w=800",
    "snow":        "https://images.pexels.com/photos/1366919/pexels-photo-1366919.jpeg?auto=compress&w=800",
    "train":       "https://images.pexels.com/photos/210607/pexels-photo-210607.jpeg?auto=compress&w=800",
    "plane":       "https://images.pexels.com/photos/358319/pexels-photo-358319.jpeg?auto=compress&w=800",
    "boat":        "https://images.pexels.com/photos/1450082/pexels-photo-1450082.jpeg?auto=compress&w=800",
    "traffic":     "https://images.pexels.com/photos/466685/pexels-photo-466685.jpeg?auto=compress&w=800",
    "fire":        "https://images.pexels.com/photos/266487/pexels-photo-266487.jpeg?auto=compress&w=800",
}

# Lazy-loaded lookup tables (built from coco_meta + videomae_meta)
_coco_url_by_text: dict[str, str] | None = None
_video_url_by_text: dict[str, str] | None = None


def _build_media_lookup() -> None:
    """Build {caption_text -> Pexels image URL} for the 5000 COCO images."""
    global _coco_url_by_text, _video_url_by_text
    if _coco_url_by_text is not None:
        return
    _coco_url_by_text = {}
    try:
        coco = pd.read_parquet("data/processed/coco_meta.parquet")
        for _, row in coco.iterrows():
            text = str(row.get("text", "")).lower()
            url = _pick_image_for_caption(text)
            if url and text:
                _coco_url_by_text[text] = url
    except Exception:
        pass
    _video_url_by_text = {}
    try:
        vid = pd.read_parquet("data/processed/videomae_meta.parquet")
        for _, row in vid.iterrows():
            label = str(row.get("label_name", ""))
            if label:
                _video_url_by_text[label.lower()] = {
                    "video_url": UCF101_VIDEO_URLS.get(label, UCF101_VIDEO_URLS["Diving"]),
                    "image_url": UCF101_THUMB_URLS.get(label, UCF101_THUMB_URLS["Diving"]),
                    "label_name": label,
                }
    except Exception:
        pass


def _pick_image_for_caption(caption: str) -> str:
    """Pick the best Pexels image URL based on the COCO caption text."""
    caption = (caption or "").lower()
    for keyword, url in COCO_KEYWORD_IMAGES.items():
        if keyword in caption:
            return url
    return "https://images.pexels.com/photos/3568520/pexels-photo-3568520.jpeg?auto=compress&w=800"


def _media_url_for_image_post(text: str) -> str:
    _build_media_lookup()
    if _coco_url_by_text is None:
        return _pick_image_for_caption(text)
    return _coco_url_by_text.get((text or "").lower()) or _pick_image_for_caption(text)


def _media_url_for_video_post(label: str) -> dict:
    _build_media_lookup()
    if _video_url_by_text is None:
        return {
            "video_url": UCF101_VIDEO_URLS.get(label, UCF101_VIDEO_URLS["Diving"]),
            "image_url": UCF101_THUMB_URLS.get(label, UCF101_THUMB_URLS["Diving"]),
        }
    return _video_url_by_text.get((label or "").lower()) or {
        "video_url": UCF101_VIDEO_URLS.get(label, UCF101_VIDEO_URLS["Diving"]),
        "image_url": UCF101_THUMB_URLS.get(label, UCF101_THUMB_URLS["Diving"]),
    }


def _media_url_for_text_post(category: str) -> str:
    return CATEGORY_IMAGE_URLS.get(category, CATEGORY_IMAGE_URLS["ENTERTAINMENT"])


def _load_resources():
    """Charge l'index FAISS et les métadonnées des posts."""
    global _index, _posts_df
    if _index is None:
        _index = faiss.read_index('data/processed/faiss_index.bin')
        _posts_df = pd.read_parquet('data/processed/huffpost_with_meta.parquet')
        print(f'FAISS index chargé : {_index.ntotal} vecteurs')


def _load_resources_v2():
    """Charge l'index FAISS v2 et les métadonnées combinées des posts."""
    global _index_v2, _posts_df_v2
    if _index_v2 is None:
        _index_v2 = faiss.read_index('data/processed/faiss_index_v2.bin')
        _index_v2.nprobe = 10
        _posts_df_v2 = pd.read_parquet('data/processed/combined_meta.parquet')
        print(f'FAISS v2 index chargé : {_index_v2.ntotal} vecteurs')

def get_feed(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20
) -> list[dict]:
    """
    Retourne le feed personnalisé pour un utilisateur.
    
    Pipeline :
    1. FAISS recherche les n_candidates posts les plus proches
    2. compute_score() calcule le score final de chacun
    3. Tri par score DESC et retour des n_results meilleurs
    
    Args:
        user_embedding: Vecteur utilisateur (384 dimensions)
        user_prefs: Préférences utilisateur (mode, interests, etc.)
        n_candidates: Nombre de candidats à évaluer
        n_results: Nombre de posts à retourner
    
    Returns:
        Liste de posts scorés, triés par pertinence
    """
    _load_resources()
    
    query = np.array([user_embedding], dtype='float32')
    sims, ids = _index.search(query, n_candidates)
    
    results = []
    user_interests = user_prefs.get('interests', [])
    
    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0 or idx >= len(_posts_df):
            continue
        
        post = _posts_df.iloc[idx].to_dict()
        
        # Convertir la date si nécessaire
        created_at = None
        if 'date' in post and pd.notna(post['date']):
            created_at = pd.Timestamp(post['date']).to_pydatetime()
        
        scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get('toxicity_score', 0.0),
            category=post.get('category', ''),
            user_interests=user_interests,
            created_at=created_at,
            user_prefs=user_prefs
        )
        
        if scored['score'] > 0:
            headline = post.get('headline', '')
            if not headline:
                headline = post.get('text', '')[:200]
            category = str(post.get('category', ''))
            results.append({
                'id': str(idx),
                'headline': headline[:200],
                'text': post.get('text', '')[:280],
                'category': category,
                'modal': 'text',
                'toxicity_score': post.get('toxicity_score', 0.0),
                'image_url': _media_url_for_text_post(category),
                'video_url': None,
                'cosine_sim': float(sim),
                'recency': scored.get('detail', {}).get('recency', 0.5),
                'score': scored['score'],
                'score_detail': scored.get('detail', {}),
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    feed = results[:n_results]
    annotate_feed(feed, user_prefs, [])
    return feed


def get_feed_v2(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20
) -> list[dict]:
    """
    Retourne le feed personnalisé v2 pour un utilisateur.
    Utilise l'index FAISS v2 et les métadonnées combinées.
    """
    _load_resources_v2()

    query = np.array([user_embedding], dtype='float32')
    sims, ids = _index_v2.search(query, n_candidates)

    results = []
    user_interests = user_prefs.get('interests', [])

    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0 or idx >= len(_posts_df_v2):
            continue

        post = _posts_df_v2.iloc[idx].to_dict()

        # Convertir la date si nécessaire
        created_at = None
        if 'date' in post and pd.notna(post['date']):
            created_at = pd.Timestamp(post['date']).to_pydatetime()

        scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get('toxicity_score', 0.0),
            category=post.get('category', ''),
            user_interests=user_interests,
            created_at=created_at,
            user_prefs=user_prefs
        )

        if scored['score'] > 0:
            headline = post.get('headline', '')
            if not headline:
                headline = post.get('text', '')[:200]
            category = str(post.get('category', ''))
            results.append({
                'id': str(idx),
                'headline': headline[:200],
                'text': post.get('text', '')[:280],
                'category': category,
                'modal': 'text',
                'toxicity_score': post.get('toxicity_score', 0.0),
                'source': post.get('source', ''),
                'image_url': _media_url_for_text_post(category),
                'video_url': None,
                'cosine_sim': float(sim),
                'recency': scored.get('detail', {}).get('recency', 0.5),
                'score': scored['score'],
                'score_detail': scored.get('detail', {}),
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    feed = results[:n_results]
    annotate_feed(feed, user_prefs, [])
    return feed


# -- FAISS V3 globals -----------------------------------------------
_index_v3 = None
_posts_df_v3 = None
_proj_384_512 = None


def _load_resources_v3() -> None:
    """Load V3 index, combined metadata, and the 384->512 projection matrix."""
    global _index_v3, _posts_df_v3, _proj_384_512
    if _index_v3 is None:
        _index_v3 = faiss.read_index("data/processed/faiss_index_v3.bin")
        _index_v3.nprobe = 10
        _posts_df_v3 = pd.read_parquet("data/processed/combined_meta_v3.parquet")
        _proj_384_512 = np.load("data/processed/proj_384_512.npy").astype("float32")
        print(f"FAISS v3 index loaded: {_index_v3.ntotal} vectors at 512-dim")


def _project_384_to_512(embedding_384: list[float]) -> np.ndarray:
    """
    Project a 384-dim MiniLM embedding into 512-dim for V3 index search.
    Used when the user_embedding was computed by MiniLM (standard path).
    """
    _load_resources_v3()
    vec = np.array(embedding_384, dtype="float32").reshape(1, 384)
    vec_512 = vec @ _proj_384_512
    norm = np.linalg.norm(vec_512)
    if norm > 0:
        vec_512 /= norm
    return vec_512.astype("float32")


def _score_post(post_dict: dict, modal: str, cosine_sim: float, user_interests: list, user_prefs: dict, row_idx: int | None = None) -> dict | None:
    """Score a post dict and return feed item dict, or None if filtered."""
    text = str(post_dict.get('text', '')) or str(post_dict.get('caption', '')) or f"{modal} content"
    created_at = None
    if 'date' in post_dict and pd.notna(post_dict.get('date')):
        created_at = pd.Timestamp(post_dict['date']).to_pydatetime()
    scored = compute_score(
        cosine_sim=cosine_sim,
        toxicity_score=0.0,
        category=post_dict.get('category', ''),
        user_interests=user_interests,
        created_at=created_at,
        user_prefs=user_prefs,
    )
    if scored['score'] <= 0:
        return None

    category = str(post_dict.get('category', 'general'))
    image_url: str | None = None
    video_url: str | None = None
    if modal == 'image':
        image_url = _media_url_for_image_post(text)
    elif modal == 'video':
        media = _media_url_for_video_post(text)
        video_url = media['video_url']
        image_url = media['image_url']
    else:
        image_url = _media_url_for_text_post(category)

    # Generate a unique ID: use explicit 'id' from data, or row_idx, or fallback to hash
    raw_id = str(post_dict.get('id', ''))
    if not raw_id or raw_id == 'nan':
        raw_id = str(row_idx or hash(text + modal))[:32]

    return {
        'id': raw_id,
        'headline': text[:200],
        'text': text[:500],
        'category': category,
        'modal': modal,
        'source': str(post_dict.get('source', 'unknown')),
        'image_url': image_url,
        'video_url': video_url,
        'cosine_sim': cosine_sim,
        'recency': scored.get('detail', {}).get('recency', 0.5),
        'score': scored['score'],
        'score_detail': scored.get('detail', {}),
    }


def _supplement_modalities(
    results: list[dict],
    df: pd.DataFrame,
    user_prefs: dict,
    n_results: int,
) -> list[dict]:
    """Supplement FAISS results with sampled posts from missing modalities."""
    content_type = user_prefs.get('content_type', 'all')
    if content_type not in ('all', 'text', 'image', 'video'):
        content_type = 'all'
    user_interests = user_prefs.get('interests', [])

    current_modals = set(r['modal'] for r in results)
    want_modals = []
    if content_type == 'all':
        if 'image' not in current_modals:
            want_modals.append('image')
        if 'video' not in current_modals:
            want_modals.append('video')
    elif content_type not in current_modals:
        want_modals.append(content_type)

    if not want_modals:
        return results

    for modal in want_modals:
        subset = df[df['modal'] == modal]
        if len(subset) == 0:
            continue
        n_sample = min(n_results, len(subset))
        sampled = subset.sample(n=n_sample, random_state=42)
        for row_idx, post in sampled.iterrows():
            post_dict = post.to_dict()
            item = _score_post(post_dict, modal, 0.5, user_interests, user_prefs, row_idx=row_idx)
            if item:
                results.append(item)

    return results


def _interleave_modalities(results: list[dict], n_results: int) -> list[dict]:
    """Round-robin interleave posts by modality."""
    by_modal: dict[str, list] = {}
    for r in results:
        by_modal.setdefault(r['modal'], []).append(r)
    # Sort each modal group by score desc
    for m in by_modal:
        by_modal[m].sort(key=lambda x: x['score'], reverse=True)

    order = [m for m in ['text', 'image', 'video'] if m in by_modal]
    interleaved = []
    max_len = max(len(by_modal[m]) for m in order) if order else 0
    for i in range(max_len):
        for m in order:
            if i < len(by_modal[m]):
                interleaved.append(by_modal[m][i])
    return interleaved[:n_results]


def _sample_diverse_feed(
    df: pd.DataFrame,
    user_prefs: dict,
    n_results: int = 20,
) -> list[dict]:
    """Sample posts evenly across modalities for new users (zero embedding)."""
    content_type = user_prefs.get('content_type', 'all')
    if content_type not in ('all', 'text', 'image', 'video'):
        content_type = 'all'
    include_images = content_type in ('all', 'image')
    include_videos = content_type in ('all', 'video')
    include_text = content_type in ('all', 'text')

    candidates = []
    user_interests = user_prefs.get('interests', [])
    sample_size = n_results * 5

    for modal, include in [('text', include_text), ('image', include_images), ('video', include_videos)]:
        if not include:
            continue
        subset = df[df['modal'] == modal]
        if len(subset) == 0:
            continue
        n_sample = min(sample_size // 3, len(subset))
        sampled = subset.sample(n=n_sample, random_state=42)
        for row_idx, post in sampled.iterrows():
            post_dict = post.to_dict()
            item = _score_post(post_dict, modal, 0.5, user_interests, user_prefs, row_idx=row_idx)
            if item:
                candidates.append(item)

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return _interleave_modalities(candidates, n_results)


def get_feed_v3(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20,
    embedding_dim: int = 384,
) -> list[dict]:
    """
    Feed multimodal v3 avec support images et vidéos.
    """
    _load_resources_v3()

    content_type = user_prefs.get('content_type', 'all')
    if content_type not in ('all', 'text', 'image', 'video'):
        content_type = 'all'
    include_images = content_type in ('all', 'image')
    include_videos = content_type in ('all', 'video')
    include_text = content_type in ('all', 'text')
    
    # Détecter embedding nul (nouvel utilisateur)
    emb = np.array(user_embedding, dtype='float32')
    emb_norm = np.linalg.norm(emb)
    is_zero_embedding = emb_norm == 0.0

    if is_zero_embedding:
        feed = _sample_diverse_feed(_posts_df_v3, user_prefs, n_results)
        annotate_feed(feed, user_prefs, [])
        return feed

    # Projeter l'embedding utilisateur
    if embedding_dim == 384 and _proj_384_512 is not None:
        query_emb = np.array([user_embedding], dtype='float32')
        query_emb = query_emb @ _proj_384_512
    else:
        query_emb = np.array([user_embedding], dtype='float32')
    
    # Normaliser
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm
    
    # Recherche FAISS
    search_k = max(n_candidates * 2, n_results * 20)
    scores, indices = _index_v3.search(query_emb, search_k)
    
    results = []
    user_interests = user_prefs.get('interests', [])
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_posts_df_v3):
            continue
        
        post = _posts_df_v3.iloc[idx].to_dict()
        modal = post.get('modal', 'text')
        
        # Filtrer par modalité selon content_type
        if modal == 'image' and not include_images:
            continue
        if modal == 'video' and not include_videos:
            continue
        if modal == 'text' and not include_text:
            continue
        
        item = _score_post(post, modal, float(score), user_interests, user_prefs, row_idx=idx)
        if item:
            results.append(item)

    # Supplement missing modalities from random sampling
    results = _supplement_modalities(results, _posts_df_v3, user_prefs, n_results)

    # Diversifier et interleaver les modalités
    if content_type == 'all' and len(results) > n_results:
        results = _interleave_modalities(results, n_results)
    elif len(results) > n_results:
        results.sort(key=lambda x: x['score'], reverse=True)

    feed = results[:n_results]
    annotate_feed(feed, user_prefs, [])
    return feed