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
        html = """<!DOCTYPE html>
<html>
<head>
    <title>🎭 Poeta Estructurado</title>
    <style>
        body {
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #f1f1f1;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.8;
        }
        
        .container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid #0f3460;
        }
        
        h1 {
            color: #e94560;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 10px rgba(233, 69, 96, 0.5);
        }
        
        .tagline {
            text-align: center;
            color: #8a8aff;
            font-style: italic;
            margin-bottom: 30px;
        }
        
        .input-group {
            margin: 20px 0;
        }
        
        input, textarea {
            width: 100%;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #0f3460;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            margin: 10px 0;
        }
        
        button {
            background: linear-gradient(45deg, #e94560, #8a8aff);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            transition: transform 0.2s;
            display: block;
            margin: 20px auto;
        }
        
        button:hover {
            transform: scale(1.05);
        }
        
        .output {
            background: rgba(15, 52, 96, 0.3);
            border-left: 4px solid #e94560;
            padding: 20px;
            margin: 20px 0;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            min-height: 100px;
        }
        
        .verse {
            padding: 15px;
            margin: 10px 0;
            background: rgba(233, 69, 96, 0.1);
            border-radius: 8px;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }
        
        .temp-slider {
            width: 200px;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            color: #8a8aff;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 POETA VERTICAL</h1>
        <div class="tagline">Generador de poesía melancólica en Chromebook</div>
        
        <div class="input-group">
            <input type="text" id="prompt" placeholder="Escribe un inicio poético... (ej: 'en la noche del alma')">
            
            <div class="controls">
                <div>
                    <label>Temperatura: <span id="temp-value">0.7</span></label>
                    <input type="range" id="temperature" class="temp-slider" min="0.1" max="1.5" step="0.1" value="0.7">
                </div>
                
                <div>
                    <label>Versos: <span id="verses-value">4</span></label>
                    <input type="range" id="verses" min="1" max="8" value="4">
                </div>
            </div>
        </div>
        
        <button onclick="generatePoetry()">✨ Generar Poesía</button>
        <button onclick="generateWithoutPrompt()" style="background: linear-gradient(45deg, #8a8aff, #0f3460);">
            🎲 Poema Aleatorio
        </button>
        
        <div id="output" class="output">
            <!-- La poesía aparecerá aquí -->
        </div>
        
        <div id="history"></div>
    </div>
    
    <footer>
        Poeta Vertical LLM · Entrenado con poemas de arachne · Chromebook Edition
    </footer>
    
    <script>
        // Actualizar valores de sliders
        document.getElementById('temperature').oninput = function() {
            document.getElementById('temp-value').textContent = this.value;
        };
        
        document.getElementById('verses').oninput = function() {
            document.getElementById('verses-value').textContent = this.value;
        };
        
        async function generatePoetry() {
            const prompt = document.getElementById('prompt').value;
            const temp = parseFloat(document.getElementById('temperature').value);
            const verses = parseInt(document.getElementById('verses').value);
            
            const output = document.getElementById('output');
            output.innerHTML = '<div class="verse">⏳ Generando poesía...</div>';
            
            try {
                const response = await fetch('http://localhost:8110/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        temperature: temp,
                        max_words: verses * 4
                    })
                });
                
                const data = await response.json();
                
                // Formatear como versos
                const words = data.generated.split(' ');
                let formatted = '';
                
                for (let i = 0; i < words.length; i++) {
                    formatted += words[i] + ' ';
                    if ((i + 1) % 4 === 0) {
                        formatted += '\\n';
                    }
                }
                
                output.innerHTML = `
                    <div class="verse">
                        <strong>Prompt:</strong> ${data.prompt || '[sin inicio]'}<br><br>
                        <strong>Poema generado:</strong><br>
                        ${formatted}
                    </div>
                `;
                
                // Añadir al historial
                addToHistory(data.prompt, formatted);
                
            } catch (error) {
                output.innerHTML = `<div class="verse">❌ Error: ${error.message}</div>`;
            }
        }
        
        function generateWithoutPrompt() {
            document.getElementById('prompt').value = '';
            generatePoetry();
        }
        
        function addToHistory(prompt, poem) {
            const history = document.getElementById('history');
            const entry = document.createElement('div');
            entry.className = 'verse';
            entry.innerHTML = `<small>${new Date().toLocaleTimeString()}</small><br>${prompt}<br>${poem}`;
            history.prepend(entry);
            
            // Limitar historial a 5 entradas
            const entries = history.getElementsByClassName('verse');
            if (entries.length > 5) {
                entries[entries.length - 1].remove();
            }
        }
        
        // Generar algo al cargar
        window.onload = function() {
            document.getElementById('prompt').value = 'el alma errante';
            setTimeout(generatePoetry, 1000);
        };
    </script>
</body>
</html>"""
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
