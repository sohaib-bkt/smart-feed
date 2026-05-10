import asyncio, sys
sys.path.insert(0, '.')
from app.db.firebase import get_db


async def clear_interactions():
    db = get_db()
    print('Suppression de toutes les interactions...')
    
    interactions_ref = db.collection("interactions")
    docs = list(interactions_ref.stream())
    
    print(f'Nombre d\'interactions à supprimer: {len(docs)}')
    
    batch = db.batch()
    for doc in docs:
        batch.delete(doc.reference)
    
    batch.commit()
    print(f'Toutes les interactions supprimées ({len(docs)} documents)')


if __name__ == '__main__':
    asyncio.run(clear_interactions())