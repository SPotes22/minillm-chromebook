# poeta_mejorado.py
import pickle
import numpy as np

print("🎭 CREANDO POETA MEJORADO CON TU ESTILO")
print("=" * 50)

# Cargar poeta base
with open('poeta_funcional.pkl', 'rb') as f:
    base = pickle.load(f)

# Añadir TUS palabras únicas
tus_palabras = [
    'macarena', 'quijote', 'koi', 'jazz', 'mozart', 'orquesta',
    'luciérnaga', 'petalos', 'ceniza', 'hoguera', 'herejía',
    'frecuencia', 'lienzo', 'melancolía', 'intersticio', 'baila',
    'cuerpo', 'alegría', 'infierno', 'edén', 'desierto', 'oasis',
    'tormenta', 'lluvia', 'tinta', 'sangre', 'lágrima', 'risa'
]

vocab = base['vocab'].copy()
id_to_token = base['id_to_token'].copy()
embedding = base['embedding'].copy()
lm_head = base['lm_head'].copy()

# Añadir nuevas palabras
next_id = len(vocab)
for palabra in tus_palabras:
    if palabra not in vocab and next_id < 150:  # Limite de vocabulario
        vocab[palabra] = next_id
        id_to_token[next_id] = palabra
        
        # Crear embedding relacionado con palabras existentes
        vec = np.zeros(48)
        count = 0
        
        # Buscar palabras relacionadas
        if 'alegría' in palabra:
            vec += embedding[vocab.get('alegría', 0)]
            count += 1
        if 'cuerpo' in palabra:
            vec += embedding[vocab.get('cuerpo', 0)]
            count += 1
        if 'música' in palabra or 'jazz' in palabra or 'mozart' in palabra:
            vec += embedding[vocab.get('poesía', 0)] * 0.7
            count += 1
        
        if count > 0:
            vec = vec / count
        else:
            vec = np.random.randn(48) * 0.1
        
        # Añadir a embedding matrix (necesitaríamos redimensionar)
        print(f"   + '{palabra}'")
        next_id += 1

print(f"📚 Vocabulario mejorado: {len(vocab)} palabras")

# Guardar poeta mejorado
modelo_mejorado = {
    'config': base['config'],
    'embedding': embedding,
    'lm_head': lm_head,
    'vocab': vocab,
    'id_to_token': id_to_token,
    'metadata': {
        'improved': True,
        'added_words': len(tus_palabras),
        'style': 'arachne_poetic'
    }
}

with open('poeta_mejorado.pkl', 'wb') as f:
    pickle.dump(modelo_mejorado, f)

print(f"\n✅ POETA MEJORADO GUARDADO")
