import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

print("🎭 POETA MEJORADO - Cargando...")
with open('poeta_mejorado.pkl', 'rb') as f:
    model = pickle.load(f)

class MejorPoet:
    def __init__(self, model):
        self.vocab = model['vocab']
        self.id_to_token = model['id_to_token']
        self.embedding = model['embedding']
        self.lm_head = model['lm_head']
        
    def generate(self, prompt, temperature=0.7, max_words=10):
        words = prompt.lower().split()
        output = []
        
        for _ in range(max_words):
            if not words:
                # Empezar con palabra aleatoria
                next_idx = np.random.randint(len(self.vocab))
            else:
                # Usar promedio de últimas 3 palabras
                last_idxs = []
                for word in words[-3:]:
                    if word in self.vocab:
                        last_idxs.append(self.vocab[word])
                
                if last_idxs:
                    # Embedding promedio
                    avg_emb = np.mean([self.embedding[idx] for idx in last_idxs], axis=0)
                    logits = avg_emb @ self.lm_head
                else:
                    # Aleatorio
                    next_idx = np.random.randint(len(self.vocab))
                    output.append(self.id_to_token[next_idx])
                    words.append(self.id_to_token[next_idx])
                    continue
            
            # Muestreo
            if temperature > 0:
                logits = logits / temperature
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                next_idx = np.random.choice(len(probs), p=probs)
            else:
                next_idx = np.argmax(logits)
            
            next_word = self.id_to_token.get(next_idx, '...')
            output.append(next_word)
            words.append(next_word)
        
        return ' '.join(output)

poet = MejorPoet(model)
print(f"✅ Listo con {len(model['vocab'])} palabras")

class PoetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        html = """
        <h1>🎭 POETA MEJORADO</h1>
        <p>Generador de poesía con estilo arachne</p>
        <p>POST /generate con JSON: {"prompt": "texto", "temperature": 0.7}</p>
        """
        self.wfile.write(html.encode())
    
    def do_POST(self):
        if self.path == '/generate':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode())
            
            prompt = data.get('prompt', '')
            temp = data.get('temperature', 0.7)
            max_words = data.get('max_words', 12)
            
            result = poet.generate(prompt, temp, max_words)
            
            response = json.dumps({
                'prompt': prompt,
                'generated': result,
                'full': prompt + (' ' + result if result else ''),
                'temperature': temp
            }, ensure_ascii=False)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(response.encode())

print("🚀 Sirviendo en http://localhost:8110")
HTTPServer(('', 8110), PoetHandler).serve_forever()
