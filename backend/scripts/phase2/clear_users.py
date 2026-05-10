import asyncio, sys
sys.path.insert(0, '.')
from app.db.firebase import get_db


async def clear_users():
    db = get_db()
    print('Suppression de tous les utilisateurs...')
    
    users_ref = db.collection("users")
    docs = list(users_ref.stream())
    
    print(f'Nombre d\'utilisateurs à supprimer: {len(docs)}')
    
    batch = db.batch()
    for doc in docs:
        batch.delete(doc.reference)
    
    batch.commit()
    print(f'Tous les utilisateurs supprimés ({len(docs)} documents)')


if __name__ == '__main__':
    asyncio.run(clear_users())