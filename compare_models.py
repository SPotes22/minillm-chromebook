# compare_models.py
import pickle
import numpy as np

print("🔍 COMPARANDO MODELOS: Viejo vs Nuevo")
print("=" * 60)

# Cargar ambos modelos
print("📂 Cargando modelos...")
with open('poeta_mejorado.pkl', 'rb') as f:
    old_model = pickle.load(f)

with open('poeta_mejorado_restored.pkl', 'rb') as f:
    new_model = pickle.load(f)

print(f"📊 Estadísticas:")
print(f"  Modelo viejo: {len(old_model['vocab'])} palabras")
print(f"  Modelo nuevo: {len(new_model['vocab'])} palabras")
print(f"  Diferencia: +{len(new_model['vocab']) - len(old_model['vocab'])} palabras")

# Función de generación común
def generate_text(prompt, model, max_words=16, temp=0.7):
    vocab = model['vocab']
    id_to_token = model['id_to_token']
    embedding = model['embedding']
    lm_head = model['lm_head']
    
    words = prompt.lower().split()
    output = []
    
    for _ in range(max_words):
        if not words:
            next_idx = np.random.randint(len(vocab))
        else:
            context_ids = []
            for word in words[-3:]:
                if word in vocab:
                    context_ids.append(vocab[word])
            
            if context_ids:
                context_emb = np.mean([embedding[idx] for idx in context_ids], axis=0)
                logits = np.dot(context_emb, lm_head)
            else:
                next_idx = np.random.randint(len(vocab))
                output.append(id_to_token[next_idx])
                words.append(id_to_token[next_idx])
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
        words.append(next_word)
    
    return ' '.join(output)

# Comparar generaciones
print("\n🎭 COMPARACIÓN DE GENERACIONES")
print("-" * 40)

test_prompts = [
    "desde el fondo de mi corazón",
    "sangre rosa",
    "el alma errante",
    "baila tu cuerpo alegria",
    "poesía de la noche"
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"  Viejo: {generate_text(prompt, old_model, 12)}")
    print(f"  Nuevo: {generate_text(prompt, new_model, 12)}")
    print()

# Análisis de vocabulario
print("\n🔠 ANÁLISIS DE VOCABULARIO")
print("-" * 40)

# Palabras en el nuevo pero no en el viejo
new_words = set(new_model['vocab'].keys()) - set(old_model['vocab'].keys())
old_words = set(old_model['vocab'].keys()) - set(new_model['vocab'].keys())

print(f"Palabras solo en nuevo: {len(new_words)}")
print(f"Palabras solo en viejo: {len(old_words)}")

print("\n📖 20 palabras nuevas añadidas:")
for i, word in enumerate(list(new_words)[:20]):
    print(f"  {i+1:2d}. {word}")

print("\n✅ Para usar el modelo restaurado:")
print("   cp poeta_mejorado_restored.pkl poeta_mejorado.pkl")
print("   python api_poeta_largo.py")