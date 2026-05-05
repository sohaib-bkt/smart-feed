"""
Aucune connexion Firebase réelle n'est nécessaire.
"""

import pytest
from app.db.mock_db import (
    reset_mock,
    get_user_profile,
    create_user,
    update_user_preferences,
    update_user_embedding,
    log_interaction,
    get_user_interactions,
)


@pytest.fixture(autouse=True)
def clean_db():
    reset_mock()
    yield
    reset_mock()


# ── Tests USERS ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_user_profile_inexistant():
    """Un user inexistant doit retourner None."""
    result = await get_user_profile("user_inconnu")
    assert result is None
    print("✅ get_user_profile(inexistant) → None")


@pytest.mark.asyncio
async def test_create_user_valeurs_par_defaut():
    """Un user créé sans prefs doit avoir les valeurs par défaut."""
    profile = await create_user("alice")

    assert profile["user_id"] == "alice"
    assert len(profile["embedding"]) == 384
    assert all(v == 0.0 for v in profile["embedding"])
    assert profile["preferences"]["mode"] == "default"
    assert profile["preferences"]["toxicity_threshold"] == 0.3
    assert profile["preferences"]["interests"] == []
    print(f"✅ create_user → profil créé : {profile['user_id']}")


@pytest.mark.asyncio
async def test_create_user_custom_preferences():
    """Un user créé avec des prefs custom doit les conserver."""
    prefs = {"mode": "focus", "interests": ["TECH", "SCIENCE"], "toxicity_threshold": 0.5}
    profile = await create_user("bob", preferences=prefs)

    assert profile["preferences"]["mode"] == "focus"
    assert "TECH" in profile["preferences"]["interests"]
    print(f"✅ create_user avec prefs custom → {profile['preferences']}")


@pytest.mark.asyncio
async def test_get_user_profile_apres_creation():
    """Après création, get_user_profile doit retourner le bon profil."""
    await create_user("charlie")
    profile = await get_user_profile("charlie")

    assert profile is not None
    assert profile["user_id"] == "charlie"
    print("✅ get_user_profile après création → OK")


@pytest.mark.asyncio
async def test_update_user_preferences():
    """Les préférences doivent être mises à jour sans toucher l'embedding."""
    await create_user("diana")
    new_prefs = {"mode": "fun", "interests": ["SPORTS"], "toxicity_threshold": 0.2}
    await update_user_preferences("diana", new_prefs)

    profile = await get_user_profile("diana")
    assert profile["preferences"]["mode"] == "fun"
    assert profile["preferences"]["interests"] == ["SPORTS"]
    # L'embedding ne doit pas avoir changé
    assert len(profile["embedding"]) == 384
    print(f"✅ update_user_preferences → mode={profile['preferences']['mode']}")


@pytest.mark.asyncio
async def test_update_user_embedding():
    """L'embedding doit être mis à jour correctement."""
    await create_user("eve")
    new_embedding = [0.1] * 384
    await update_user_embedding("eve", new_embedding)

    profile = await get_user_profile("eve")
    assert profile["embedding"][0] == pytest.approx(0.1)
    assert len(profile["embedding"]) == 384
    print("✅ update_user_embedding → embedding mis à jour")


@pytest.mark.asyncio
async def test_update_preferences_user_inexistant():
    """Mettre à jour un user inexistant doit lever une erreur."""
    with pytest.raises(KeyError):
        await update_user_preferences("fantome", {"mode": "fun"})
    print("✅ update_preferences sur user inexistant → KeyError levée")


# ── Tests INTERACTIONS ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_interaction_simple():
    """Une interaction loguée doit être récupérable."""
    await log_interaction({"user_id": "frank", "post_id": "post_1", "action": "like"})
    interactions = await get_user_interactions("frank")

    assert len(interactions) == 1
    assert interactions[0]["action"] == "like"
    assert interactions[0]["post_id"] == "post_1"
    assert "timestamp" in interactions[0]
    print("✅ log_interaction + get_user_interactions → OK")


@pytest.mark.asyncio
async def test_interactions_isolees_par_user():
    """Les interactions de deux users ne doivent pas se mélanger."""
    await log_interaction({"user_id": "grace", "post_id": "p1", "action": "like"})
    await log_interaction({"user_id": "grace", "post_id": "p2", "action": "skip"})
    await log_interaction({"user_id": "henry", "post_id": "p3", "action": "like"})

    grace_interactions = await get_user_interactions("grace")
    henry_interactions = await get_user_interactions("henry")

    assert len(grace_interactions) == 2
    assert len(henry_interactions) == 1
    print(f"✅ Isolation par user → grace={len(grace_interactions)}, henry={len(henry_interactions)}")


@pytest.mark.asyncio
async def test_interactions_ordre_decroissant():
    """Les interactions doivent être retournées du plus récent au plus ancien."""
    import asyncio
    await log_interaction({"user_id": "ivan", "post_id": "p1", "action": "like"})
    await asyncio.sleep(0.01)  # petit délai pour garantir des timestamps différents
    await log_interaction({"user_id": "ivan", "post_id": "p2", "action": "watch_full"})

    interactions = await get_user_interactions("ivan")
    assert interactions[0]["post_id"] == "p2"  # la plus récente en premier
    assert interactions[1]["post_id"] == "p1"
    print("✅ Ordre décroissant par timestamp → OK")


@pytest.mark.asyncio
async def test_get_interactions_user_vide():
    """Un user sans interactions doit retourner une liste vide."""
    interactions = await get_user_interactions("nobody")
    assert interactions == []
    print("✅ get_user_interactions(user vide) → []")


@pytest.mark.asyncio
async def test_last_n_limit():
    """Le paramètre last_n doit limiter le nombre de résultats."""
    for i in range(10):
        await log_interaction({"user_id": "julia", "post_id": f"p{i}", "action": "like"})

    interactions = await get_user_interactions("julia", last_n=3)
    assert len(interactions) == 3
    print(f"✅ last_n=3 sur 10 interactions → {len(interactions)} résultats")


if __name__ == "__main__":
    import asyncio

    async def run_all():
        clean_db_fixture = clean_db()
        next(clean_db_fixture)

        await test_get_user_profile_inexistant()
        await test_create_user_valeurs_par_defaut()
        await test_create_user_custom_preferences()
        await test_get_user_profile_apres_creation()
        await test_update_user_preferences()
        await test_update_user_embedding()
        await test_update_preferences_user_inexistant()
        await test_log_interaction_simple()
        await test_interactions_isolees_par_user()
        await test_interactions_ordre_decroissant()
        await test_get_interactions_user_vide()
        await test_last_n_limit()
        print("\n✅ Tous les tests Firebase (mock) sont passés !")

    asyncio.run(run_all())