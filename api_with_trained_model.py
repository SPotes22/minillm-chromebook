"""
API para Vertical LLM con modelo entrenado
Versión optimizada para el modelo transferido desde PyTorch
"""

import numpy as np
import pickle
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os

# Cargar modelo entrenado
print("📦 Cargando modelo entrenado...")
with open('vertical_model_numpy.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Configuración
class VerticalConfig:
    def __init__(self, vocab_size, d_model, n_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

config = VerticalConfig(**model_data['config'])

class VerticalBlock:
    def __init__(self):
        self.W_state = None
        self.b_state = None
        self.W_out = None
        self.b_out = None
    
    def forward(self, x, state):
        combined = np.concatenate([x, state])
        new_state = np.tanh(combined @ self.W_state + self.b_state)
        output = new_state @ self.W_out + self.b_out
        return new_state, output

class TrainedVerticalLLM:
    def __init__(self):
        self.config = config
        self.embedding = model_data['embedding']
        self.lm_head = model_data['lm_head']
        self.b_head = model_data['b_head']
        
        # Bloques
        self.blocks = []
        for block_data in model_data['blocks']:
            block = VerticalBlock()
            block.W_state = block_data['W_state']
            block.b_state = block_data['b_state']
            block.W_out = block_data['W_out']
            block.b_out = block_data['b_out']
            self.blocks.append(block)
        
        # Tokenizador
        self.vocab = model_data['vocab']
        self.id_to_token = model_data['id_to_token']
        
        # Estado
        self.state = np.zeros(config.d_model, dtype=np.float32)
        self.position = 0
    
    def encode(self, text):
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get('<UNK>', 0))
        return tokens
    
    def decode(self, tokens):
        return ''.join([self.id_to_token.get(t, '') for t in tokens])
    
    def reset(self):
        self.state = np.zeros(config.d_model, dtype=np.float32)
        self.position = 0
    
    def generate(self, prompt, max_tokens=50, temperature=0.7):
        self.reset()
        tokens = self.encode(prompt)
        
        # Procesar prompt
        for token in tokens:
            x = self.embedding[token]
            current_state = self.state
            for block in self.blocks:
                current_state, x = block.forward(x, current_state)
            self.state = current_state
        
        # Generar
        generated = []
        for _ in range(max_tokens):
            # Último token
            last_token = tokens[-1] if not generated else generated[-1]
            x = self.embedding[last_token]
            
            current_state = self.state
            for block in self.blocks:
                current_state, x = block.forward(x, current_state)
            self.state = current_state
            
            # Logits
            logits = x @ self.lm_head + self.b_head
            
            # Muestreo
            if temperature > 0:
                logits = logits / temperature
                logits = logits - np.max(logits)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(logits)
            
            generated.append(int(next_token))
            
            if next_token == self.vocab.get('<EOS>', 1):
                break
        
        text = self.decode(generated)
        return {
            "prompt": prompt,
            "generated": text,
            "full": prompt + text
        }

# Crear modelo
model = TrainedVerticalLLM()
print(f"✅ Modelo cargado: {config.vocab_size} tokens, {config.d_model} dim, {config.n_layers} capas")

# API Server
class LLMAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Vertical LLM API (Trained)</h1><p>Use POST /generate</p>')
        elif self.path == '/status':
            self.send_json({
                "status": "online",
                "model": "TrainedVerticalLLM",
                "vocab_size": config.vocab_size,
                "d_model": config.d_model,
                "n_layers": config.n_layers
            })
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 50)
            temperature = data.get('temperature', 0.7)
            
            result = model.generate(prompt, max_tokens, temperature)
            
            self.send_json(result)
        else:
            self.send_error(404)
    
    def send_json(self, data):
        response = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

def run_server(port=8080):
    server = HTTPServer(('', port), LLMAPIHandler)
    print(f"🚀 API iniciada en http://localhost:{port}")
    print("📡 Endpoint: POST /generate")
    server.serve_forever()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    
    run_server(args.port)