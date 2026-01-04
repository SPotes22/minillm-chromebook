# restore_original_model.py
import pickle
import numpy as np

print("🔄 RESTAURANDO MODELO ORIGINAL CON VOCABULARIO AMPLIADO")
print("=" * 60)

# Cargar el modelo original que sabemos que funciona
with open('poeta_funcional.pkl', 'rb') as f:
    base_model = pickle.load(f)

print(f"📦 Modelo base cargado: {len(base_model['vocab'])} palabras")

# Vocabulario adicional de tus textos
additional_words = [
    # Palabras de tus poemas que no están en el vocabulario original
    'cabellos', 'castaños', 'cuerpo', 'alegria', 'macarena', 'infierno',
    'quijote', 'koi', 'edén', 'jazz', 'mozart', 'orquesta', 'desierto',
    'oasis', 'tormenta', 'luciérnaga', 'petalos', 'ceniza', 'hoguera', 
    'herejía', 'frecuencia', 'lienzo', 'melancolía', 'intersticio', 'baila',
    'alegría', 'edén', 'tinta', 'sangre', 'lágrima', 'risa', 'placer',
    'esencia', 'dura', 'dulce', 'pura', 'arder', 'espero', 'manos',
    'abiertas', 'natural', 'ahoga', 'islas', 'calor', 'playas', 'concreto',
    'naranja', 'atardecer', 'sequía', 'perecemos', 'nilo', 'agotado',
    'culminó', 'caímos', 'vueltas', 'cruces', 'clave', 'sol', 'plantas',
    'polo', 'tierra', 'arrastrando', 'mareas', 'atormente', 'claro',
    'luna', 'asemeje', 'salvación', 'místico', 'final', 'costó',
    'volviere', 'cambiaría', 'secreto', 'caminar', 'auténtico', 'diablo',
    'guste', 'dejan', 'frutos', 'canta', 'toca', 'guitarra', 'patas',
    'encerrado', 'historia', 'salgan', 'sílaba', 'cantaré', 'diré',
    'locos', 'vernos', 'eternos', 'dimensión', 'conocen', 'encontraré',
    'perfecta', 'estrellaré', 'faceta', 'simétricos', 'izquierda', 'derecha',
    'paré', 'propósito', 'dejé', 'familia', 'amigos', 'tierra', 'promesas',
    'falsas', 'esperanza', 'exacta', 'segundos', 'contados', 'cuello',
    'dije', 'lleve', 'rojo', 'ascender', 'hormiga', 'pared', 'reflejo',
    'eterna', 'sentía', 'veintiséis', 'veintisiete', 'devolvieron',
    'milésimas', 'obtuve', 'conocimiento', 'bendito', 'secretos', 'merecemos',
    'inclusive', 'nadara', 'espaldas', 'inundada', 'esperaba', 'detiene',
    'recuerdo', 'sonrisa', 'jocosa', 'caricias', 'sutiles', 'hierve',
    'sal', 'eché', 'limpiar', 'sacafuera', 'digno', 'tocarte', 'canción',
    'compondría', 'necedad', 'locura', 'alejaba', 'sensibilidad', 'controlar',
    'emociones', 'confundirías', 'respondió', 'pétalos', 'suaves', 'coloridos',
    'dependían', 'arranqué', 'hermosa', 'completa', 'realización', 'belleza',
    'compone', 'blanca', 'radiara', 'piel', 'saben', 'hecha', 'tomó',
    'energía', 'transforma', 'liberador', 'complejo', 'solitario', 'vacío',
    'hoguera', 'acabado', 'yacían', 'motivación', 'disuelve', 'inherente',
    'placer', 'verte', 'sigue', 'latiendo', 'estremece', 'suspira', 'tardes',
    'verano', 'noches', 'invernales', 'meditar', 'grande', 'universo',
    'insignificante', 'parece', 'herejía', 'tartamudea', 'pensamientos',
    'rayo', 'ecuación', 'física', 'explica', 'realidad', 'pensaba',
    'faltó', 'conocerme', 'constante', 'cambio', 'arrastrarme', 'coraje',
    'caminando', 'hojas', 'blanco', 'lluvia', 'sentido', 'coherencia',
    'nuevos', 'tirando', 'ensimismado', 'obra', 'volvía', 'hermoso',
    'arrepentimiento', 'contados', 'salvando', 'ahogarme', 'paradójicamente',
    'alimentando'
]

# Expandir el vocabulario
vocab = base_model['vocab'].copy()
id_to_token = base_model['id_to_token'].copy()

next_id = len(vocab)
for word in additional_words:
    if word not in vocab and len(word) > 1:  # Solo palabras de más de 1 letra
        vocab[word] = next_id
        id_to_token[next_id] = word
        next_id += 1

print(f"📚 Vocabulario ampliado: {len(vocab)} palabras")
print(f"   (+{len(vocab) - len(base_model['vocab'])} palabras nuevas)")

# Expandir embeddings y lm_head manteniendo la estructura original
embedding_dim = base_model['embedding'].shape[1]
old_vocab_size = base_model['embedding'].shape[0]
new_vocab_size = len(vocab)

# Crear nuevas matrices ampliadas
new_embedding = np.zeros((new_vocab_size, embedding_dim), dtype=np.float32)
new_lm_head = np.zeros((embedding_dim, new_vocab_size), dtype=np.float32)

# Copiar valores originales
new_embedding[:old_vocab_size] = base_model['embedding']
new_lm_head[:, :old_vocab_size] = base_model['lm_head']

# Inicializar nuevas palabras con embeddings de palabras similares
print("\n🎨 Inicializando nuevas palabras...")

for word, idx in vocab.items():
    if idx >= old_vocab_size:  # Es una palabra nueva
        # Buscar palabra similar en el vocabulario original
        similar_words = []
        
        # Buscar por prefijo/sufijo
        for old_word, old_idx in base_model['vocab'].items():
            if (word[:3] in old_word or old_word[:3] in word or
                word[-3:] in old_word or old_word[-3:] in word):
                similar_words.append(old_idx)
        
        if similar_words:
            # Promedio de embeddings similares
            avg_embedding = np.mean([base_model['embedding'][i] for i in similar_words[:3]], axis=0)
            new_embedding[idx] = avg_embedding
            
            # Para lm_head, usar promedio de las columnas correspondientes
            avg_lm_col = np.mean([base_model['lm_head'][:, i] for i in similar_words[:3]], axis=0)
            new_lm_head[:, idx] = avg_lm_col
            print(f"   ✓ '{word}' inicializada con palabras similares")
        else:
            # Inicialización aleatoria suave
            new_embedding[idx] = np.random.randn(embedding_dim) * 0.05
            new_lm_head[:, idx] = np.random.randn(embedding_dim) * 0.05
            print(f"   ✗ '{word}' inicializada aleatoriamente")

# Guardar modelo mejorado
improved_model = {
    'config': base_model['config'],
    'embedding': new_embedding,
    'lm_head': new_lm_head,
    'vocab': vocab,
    'id_to_token': id_to_token,
    'blocks': base_model.get('blocks', []),
    'metadata': {
        'improved': True,
        'original_words': old_vocab_size,
        'added_words': new_vocab_size - old_vocab_size,
        'total_words': new_vocab_size,
        'description': 'Modelo original ampliado con vocabulario de poemas'
    }
}

with open('poeta_mejorado_restored.pkl', 'wb') as f:
    pickle.dump(improved_model, f)

print(f"\n✅ MODELO RESTAURADO GUARDADO: poeta_mejorado_restored.pkl")
print(f"   • Palabras totales: {new_vocab_size}")
print(f"   • Dimensión: {embedding_dim}D")

# Prueba rápida
print("\n🧪 Prueba de coherencia:")
print("-" * 40)

def test_generation(prompt, vocab, id_to_token, embedding, lm_head, temp=0.7, words=8):
    """Función de generación simple para prueba"""
    prompt_words = prompt.lower().split()
    output = []
    
    for _ in range(words):
        if not prompt_words:
            next_idx = np.random.randint(len(vocab))
        else:
            context_ids = []
            for word in prompt_words[-3:]:
                if word in vocab:
                    context_ids.append(vocab[word])
            
            if context_ids:
                context_emb = np.mean([embedding[idx] for idx in context_ids], axis=0)
                logits = np.dot(context_emb, lm_head)
            else:
                next_idx = np.random.randint(len(vocab))
                output.append(id_to_token[next_idx])
                prompt_words.append(id_to_token[next_idx])
                continue
        
        if temp > 0:
            logits = logits / temp
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            next_idx = np.random.choice(len(probs), p=probs)
        else:
            next_idx = np.argmax(logits)
        
        next_word = id_to_token.get(next_idx, '...')
        output.append(next_word)
        prompt_words.append(next_word)
    
    return ' '.join(output)

# Probar con prompts significativos
test_prompts = [
    "desde el fondo de mi corazón",
    "sangre rosa",
    "el alma errante",
    "baila tu cuerpo"
]

for prompt in test_prompts:
    result = test_generation(prompt, vocab, id_to_token, new_embedding, new_lm_head, 
                           temp=0.7, words=8)
    print(f"'{prompt}' → {result}")