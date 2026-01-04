"""
poeta_from_scratch.py - Crear poeta funcional desde cero
"""

import numpy as np
import pickle
import json

print("🎭 CREANDO POETA FUNCIONAL DESDE CERO")
print("=" * 50)

# 1. VOCABULARIO SENSATO
vocab = {}
id_to_token = {}

# Palabras de TUS poemas
poem_words = [
    # De tus poemas
    'amor', 'vida', 'muerte', 'alma', 'corazón', 'poesía', 'verso',
    'noche', 'día', 'tiempo', 'universo', 'luz', 'sombra', 'mar',
    'cielo', 'tierra', 'fuego', 'agua', 'aire', 'sangre', 'lágrima',
    'risa', 'dolor', 'alegría', 'tristeza', 'esperanza', 'memoria',
    'sueño', 'realidad', 'fantasía', 'verdad', 'mentira', 'camino',
    'destino', 'azar', 'libertad', 'prisión', 'guerra', 'paz',
    
    # Tu vocabulario personal
    'cabellos', 'castaños', 'cuerpo', 'alegria', 'macarena', 'infierno',
    'quijote', 'koi', 'edén', 'jazz', 'mozart', 'orquesta', 'desierto',
    'oasis', 'tormenta', 'luciérnaga', 'petalos', 'rosa', 'ceniza',
    'hoguera', 'herejía', 'frecuencia', 'lienzo', 'melancolía',
    
    # Palabras comunes
    'el', 'la', 'los', 'las', 'un', 'una', 'y', 'de', 'en', 'con',
    'por', 'para', 'sin', 'sobre', 'entre', 'hacia', 'desde', 'que',
    'como', 'cuando', 'donde', 'porque', 'si', 'no', 'sí', 'también',
    'muy', 'más', 'menos', 'todo', 'nada', 'algo', 'siempre', 'nunca',
    'ahora', 'después', 'antes', 'aquí', 'allí', 'lejos', 'cerca',
    
    # Verbos
    'es', 'era', 'soy', 'eres', 'somos', 'son', 'está', 'estaba',
    'tengo', 'tiene', 'quiero', 'puedo', 'debo', 'sé', 'sabe',
    'ama', 'odia', 'vive', 'muere', 'nace', 'crece', 'cambia',
    'escribe', 'lee', 'canta', 'baila', 'corre', 'salta', 'vuela',
    'piensa', 'siente', 'recuerda', 'olvida', 'encuentra', 'pierde',
    
    # Inglés básico
    'hello', 'world', 'love', 'life', 'death', 'soul', 'heart',
    'poetry', 'verse', 'night', 'day', 'time', 'light', 'shadow',
    'sea', 'sky', 'earth', 'fire', 'water', 'air', 'blood', 'tear'
]

# Crear mapeos
for i, word in enumerate(poem_words[:100]):  # Máximo 100 palabras
    vocab[word] = i
    id_to_token[i] = word

print(f"📚 Vocabulario creado: {len(vocab)} palabras")
print("Primeras 10:", list(vocab.keys())[:10])

# 2. EMBEDDINGS CON SENTIDO
vocab_size = len(vocab)
d_model = 48

print(f"\n🎨 Creando embeddings inteligentes...")

# Crear espacio semántico
embedding = np.zeros((vocab_size, d_model), dtype=np.float32)

# Semillas para dimensiones semánticas
np.random.seed(42)
semantic_axes = np.random.randn(10, d_model)  # 10 ejes semánticos

for word, idx in vocab.items():
    # Vector base aleatorio
    vec = np.random.randn(d_model) * 0.1
    
    # Añadir significado según la palabra
    if word in ['amor', 'love', 'cariño', 'affection']:
        vec += semantic_axes[0] * 0.5  # Eje amor
    elif word in ['vida', 'life', 'existencia']:
        vec += semantic_axes[1] * 0.5  # Eje vida
    elif word in ['muerte', 'death', 'fin']:
        vec += semantic_axes[2] * 0.5  # Eje muerte
    elif word in ['poesía', 'poetry', 'verso', 'verse']:
        vec += semantic_axes[3] * 0.5  # Eje poesía
    elif word in ['noche', 'night', 'oscuridad']:
        vec += semantic_axes[4] * 0.3  # Eje noche
    elif word in ['luz', 'light', 'brillo']:
        vec += semantic_axes[5] * 0.3  # Eje luz
    elif word in ['tristeza', 'sadness', 'dolor', 'pain']:
        vec += semantic_axes[6] * 0.4  # Eje tristeza
    elif word in ['alegría', 'joy', 'felicidad']:
        vec += semantic_axes[7] * 0.4  # Eje alegría
    
    # Palabras relacionadas tienen embeddings similares
    if word in ['cabellos', 'pelo', 'melena']:
        vec = embedding[vocab.get('cuerpo', 0)] * 0.8 + np.random.randn(d_model) * 0.1
    
    embedding[idx] = vec

print(f"   Embedding shape: {embedding.shape}")

# 3. LM HEAD INTELIGENTE
print(f"\n🧠 Creando LM head...")
lm_head = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.1

# Hacer que palabras relacionadas tengan probabilidades similares
for i in range(vocab_size):
    word1 = id_to_token[i]
    
    # Buscar palabras relacionadas
    for j in range(vocab_size):
        if i == j:
            continue
            
        word2 = id_to_token[j]
        
        # Si son sinónimos o relacionados
        related = False
        related_pairs = [
            ('amor', 'love'), ('vida', 'life'), ('muerte', 'death'),
            ('poesía', 'poetry'), ('noche', 'night'), ('luz', 'light'),
            ('cabellos', 'pelo'), ('cuerpo', 'body')
        ]
        
        for pair in related_pairs:
            if (word1 == pair[0] and word2 == pair[1]) or (word1 == pair[1] and word2 == pair[0]):
                related = True
                break
        
        if related:
            # Hacer sus logits similares
            lm_head[:, i] = lm_head[:, i] * 0.7 + lm_head[:, j] * 0.3

# 4. CONFIGURACIÓN
config = {
    'vocab_size': vocab_size,
    'd_model': d_model,
    'n_layers': 3,
    'is_poet': True,
    'author': 'arachne',
    'created': 'from_scratch'
}

# 5. GUARDAR
model_data = {
    'config': config,
    'embedding': embedding,
    'lm_head': lm_head,
    'vocab': vocab,
    'id_to_token': id_to_token,
    'blocks': [],  # Para compatibilidad
    'metadata': {
        'type': 'functional_poet',
        'words': list(vocab.keys()),
        'note': 'Poeta creado desde cero con vocabulario sensato'
    }
}

output_file = 'poeta_funcional.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ POETA FUNCIONAL CREADO: {output_file}")
print(f"   • Palabras: {len(vocab)}")
print(f"   • Dimensiones: {d_model}D")
print(f"   • Embedding shape: {embedding.shape}")

# 6. PRUEBA RÁPIDA
print(f"\n🧪 Prueba rápida:")
test_words = ['amor', 'vida', 'poesía', 'cabellos', 'hello']
for word in test_words:
    if word in vocab:
        idx = vocab[word]
        vec = embedding[idx]
        print(f"   '{word}' (ID {idx}): norma={np.linalg.norm(vec):.3f}")
    else:
        print(f"   '{word}': ❌ No en vocabulario")
