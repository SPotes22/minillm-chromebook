"""
Mini-LLM Transformer optimizado para Chromebook
Versión 2.0 - Completamente funcional
"""

import numpy as np
import json
import time
import pickle
import os
import sys
from typing import List, Tuple, Dict, Optional

class Config:
    """Configuración del modelo"""
    def __init__(self, vocab_size=100, d_model=128, n_heads=4, 
                 n_layers=3, d_ff=512, max_seq_len=128, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.d_head = d_model // n_heads
        
class RotaryPositionalEncoding:
    """Codificación posicional RoPE - VERSIÓN SIMPLIFICADA Y FUNCIONAL"""
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        
    def apply_rope(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """Aplica RoPE de manera simplificada pero funcional"""
        batch_size, seq_len, n_heads, d_head = x.shape
        
        # Crear frecuencias para las posiciones
        positions = np.arange(start_pos, start_pos + seq_len).reshape(1, seq_len, 1, 1)
        indices = np.arange(0, d_head, 2).reshape(1, 1, 1, -1)
        
        # Calcular frecuencias
        theta = 1.0 / (10000 ** (indices / d_head))
        freqs = positions * theta
        
        # Calcular cos y sin
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        
        # Separar dimensiones pares e impares
        x_reshaped = x.reshape(batch_size, seq_len, n_heads, d_head // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        
        # Aplicar rotación
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        # Reconstruir
        x_rot = np.stack([x_rot_even, x_rot_odd], axis=-1)
        return x_rot.reshape(batch_size, seq_len, n_heads, d_head)

class MemoryEfficientAttention:
    """Atención optimizada para memoria - VERSIÓN SIMPLIFICADA"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        
        # Pesos - inicialización más pequeña
        scale = 0.1 / np.sqrt(d_model)
        self.Wq = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wk = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wv = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.Wo = np.random.randn(d_model, d_model).astype(np.float32) * scale
        
    def _stable_softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax numéricamente estable"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, x: np.ndarray, rope: RotaryPositionalEncoding, 
                start_pos: int = 0) -> np.ndarray:
        """Forward pass simplificado pero funcional"""
        # Asegurar que x sea 3D
        if x.ndim == 2:
            x = x.reshape(1, -1, self.d_model)
        
        batch_size, seq_len, _ = x.shape
        
        # Proyecciones lineales
        Q = np.matmul(x, self.Wq.T).reshape(batch_size, seq_len, self.n_heads, self.d_head)
        K = np.matmul(x, self.Wk.T).reshape(batch_size, seq_len, self.n_heads, self.d_head)
        V = np.matmul(x, self.Wv.T).reshape(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Aplicar RoPE
        Q = rope.apply_rope(Q, start_pos)
        K = rope.apply_rope(K, start_pos)
        
        # Transponer para atención multi-head
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, d_head]
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Calcular scores de atención
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        
        # Máscara causal
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = scores - 1e9 * mask[None, None, :, :]
        
        # Softmax
        attn_weights = self._stable_softmax(scores, axis=-1)
        
        # Aplicar a valores
        context = np.matmul(attn_weights, V)
        
        # Recombinar cabezas
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Proyección final
        return np.matmul(context, self.Wo.T)

class FeedForward:
    """FFN simple sin SwiGLU (para simplificar)"""
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        scale = 0.1 / np.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
        self.b1 = np.zeros(d_ff).astype(np.float32)
        self.b2 = np.zeros(d_model).astype(np.float32)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Capa intermedia con GELU aproximado
        hidden = np.matmul(x, self.W1) + self.b1
        # GELU aproximado: x * sigmoid(1.702 * x)
        hidden = hidden * (1 / (1 + np.exp(-1.702 * hidden)))
        return np.matmul(hidden, self.W2) + self.b2

class LayerNorm:
    """LayerNorm simplificado"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Asegurar que x sea 2D o 3D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

class TransformerBlock:
    """Bloque transformer simplificado"""
    def __init__(self, config: Config):
        self.attention = MemoryEfficientAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.ffn = FeedForward(config.d_model, config.d_ff)
        self.ln1 = LayerNorm(config.d_model)
        self.ln2 = LayerNorm(config.d_model)
        
    def forward(self, x: np.ndarray, rope: RotaryPositionalEncoding, 
                start_pos: int = 0) -> np.ndarray:
        # Atención con residual y pre-norm
        norm_x = self.ln1.forward(x)
        attn_out = self.attention.forward(norm_x, rope, start_pos)
        x = x + attn_out
        
        # FFN con residual y pre-norm
        norm_x = self.ln2.forward(x)
        ffn_out = self.ffn.forward(norm_x)
        x = x + ffn_out
        
        return x

class MiniLLM:
    """Modelo principal mini-LLM - VERSIÓN FUNCIONAL"""
    def __init__(self, config: Config):
        self.config = config
        self.rope = RotaryPositionalEncoding(config.d_head)
        
        # Inicialización de parámetros
        scale = 0.1 / np.sqrt(config.d_model)
        
        # Embeddings
        self.token_embedding = np.random.randn(
            config.vocab_size, config.d_model
        ).astype(np.float32) * scale
        
        # Capas transformer
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        
        # Normalización final y cabeza LM
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = np.random.randn(
            config.d_model, config.vocab_size
        ).astype(np.float32) * scale
        
        self.current_pos = 0
    
    def _ensure_3d(self, x: np.ndarray) -> np.ndarray:
        """Asegura que el tensor sea 3D"""
        if x.ndim == 1:
            return x.reshape(1, 1, -1)
        elif x.ndim == 2:
            return x.reshape(1, x.shape[0], x.shape[1])
        return x
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax estable"""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _sample(self, logits: np.ndarray, temperature: float = 1.0, 
                top_p: float = 0.9) -> int:
        """Muestreo con temperatura y top-p"""
        if temperature > 0:
            logits = logits / temperature
            probs = self._softmax(logits)
            
            # Top-p (nucleus) sampling
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Eliminar tokens con probabilidad acumulada > top_p
            mask = cumulative_probs <= top_p
            # Siempre mantener al menos un token
            if not np.any(mask):
                mask[0] = True
            
            filtered_indices = sorted_indices[mask]
            filtered_probs = sorted_probs[mask]
            filtered_probs = filtered_probs / np.sum(filtered_probs)
            
            return int(np.random.choice(filtered_indices, p=filtered_probs))
        else:
            return int(np.argmax(logits))
    
    def generate(self, prompt: List[int], max_tokens: int = 20, 
                temperature: float = 0.8, top_p: float = 0.9) -> List[int]:
        """Generación de texto - VERSIÓN FUNCIONAL"""
        tokens = list(prompt)  # Copia
        self.current_pos = 0
        
        for i in range(max_tokens):
            # Preparar entrada para este paso
            if i == 0:
                # Primer paso: usar todo el prompt
                input_tokens = tokens
                start_pos = 0
            else:
                # Pasos siguientes: usar solo el último token
                input_tokens = tokens[-1:]
                start_pos = len(tokens) - 1
            
            # Obtener embeddings
            x = self.token_embedding[input_tokens]
            x = self._ensure_3d(x)  # [1, seq_len, d_model]
            
            # Forward pass a través de las capas
            for layer in self.layers:
                x = layer.forward(x, self.rope, start_pos)
            
            # Normalización final
            x = self.ln_f.forward(x)
            
            # Logits del último token
            last_token_embedding = x[0, -1, :]  # [d_model]
            logits = np.matmul(last_token_embedding, self.lm_head)  # [vocab_size]
            
            # Muestreo del siguiente token
            next_token = self._sample(logits, temperature, top_p)
            
            # Agregar token a la secuencia
            tokens.append(next_token)
            
            # Condiciones de parada simples
            if next_token == 0:  # Token especial de fin
                break
            if len(tokens) >= 2 and tokens[-1] == tokens[-2]:
                break  # Evitar repetición infinita
        
        return tokens
    
    def save(self, path: str):
        """Guardar modelo"""
        data = {
            'config': self.config.__dict__,
            'token_embedding': self.token_embedding,
            'lm_head': self.lm_head,
        }
        
        # Guardar pesos de cada capa
        layers_data = []
        for i, layer in enumerate(self.layers):
            layer_data = {
                'Wq': layer.attention.Wq,
                'Wk': layer.attention.Wk,
                'Wv': layer.attention.Wv,
                'Wo': layer.attention.Wo,
                'W1': layer.ffn.W1,
                'W2': layer.ffn.W2,
                'b1': layer.ffn.b1,
                'b2': layer.ffn.b2,
                'ln1_gamma': layer.ln1.gamma,
                'ln1_beta': layer.ln1.beta,
                'ln2_gamma': layer.ln2.gamma,
                'ln2_beta': layer.ln2.beta,
            }
            layers_data.append(layer_data)
        
        data['layers'] = layers_data
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Modelo guardado en {path}")
    
    @classmethod
    def load(cls, path: str):
        """Cargar modelo"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Crear configuración
        config_dict = data['config']
        config = Config(
            vocab_size=config_dict['vocab_size'],
            d_model=config_dict['d_model'],
            n_heads=config_dict['n_heads'],
            n_layers=config_dict['n_layers'],
            d_ff=config_dict['d_ff'],
            max_seq_len=config_dict['max_seq_len'],
            dropout=config_dict['dropout']
        )
        
        # Crear modelo
        model = cls(config)
        model.token_embedding = data['token_embedding']
        model.lm_head = data['lm_head']
        
        # Cargar pesos de cada capa
        for i, layer_data in enumerate(data['layers']):
            model.layers[i].attention.Wq = layer_data['Wq']
            model.layers[i].attention.Wk = layer_data['Wk']
            model.layers[i].attention.Wv = layer_data['Wv']
            model.layers[i].attention.Wo = layer_data['Wo']
            model.layers[i].ffn.W1 = layer_data['W1']
            model.layers[i].ffn.W2 = layer_data['W2']
            model.layers[i].ffn.b1 = layer_data['b1']
            model.layers[i].ffn.b2 = layer_data['b2']
            model.layers[i].ln1.gamma = layer_data['ln1_gamma']
            model.layers[i].ln1.beta = layer_data['ln1_beta']
            model.layers[i].ln2.gamma = layer_data['ln2_gamma']
            model.layers[i].ln2.beta = layer_data['ln2_beta']
        
        print(f"✅ Modelo cargado desde {path}")
        return model

class SimpleTokenizer:
    """Tokenizador simple pero funcional"""
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        
        # Crear vocabulario básico
        self.vocab = {}
        self.id_to_token = {}
        
        # Caracteres ASCII básicos
        chars = []
        chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])  # a-z
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])  # A-Z
        chars.extend([chr(i) for i in range(ord('0'), ord('9') + 1)])  # 0-9
        chars.extend([' ', '.', ',', '!', '?', '\n', '"', "'", '-', ':', ';'])
        
        # Limitar al tamaño del vocabulario
        chars = chars[:min(vocab_size - 4, len(chars))]
        
        # Tokens regulares
        for i, char in enumerate(chars):
            self.vocab[char] = i
            self.id_to_token[i] = char
        
        # Tokens especiales
        special_tokens = ['<PAD>', '<EOS>', '<UNK>', '<BOS>']
        for i, token in enumerate(special_tokens, start=len(chars)):
            self.vocab[token] = i
            self.id_to_token[i] = token
    
    def encode(self, text: str) -> List[int]:
        """Codificar texto a tokens"""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get('<UNK>', 0))
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decodificar tokens a texto"""
        return ''.join([self.id_to_token.get(token, '?') for token in tokens])

def interactive_demo():
    """Demo interactiva simple"""
    print("🚀 Mini-LLM para Chromebook - Demo Interactiva")
    print("=" * 50)
    
    # Configuración minimal
    config = Config(
        vocab_size=100,
        d_model=64,  # Más pequeño para Chromebook
        n_heads=4,
        n_layers=2,  # Menos capas
        d_ff=256,
        max_seq_len=64
    )
    
    # Inicializar modelo y tokenizador
    model_path = "minillm_model.pkl"
    
    if os.path.exists(model_path):
        print("📂 Cargando modelo existente...")
        model = MiniLLM.load(model_path)
    else:
        print("🆕 Creando nuevo modelo...")
        model = MiniLLM(config)
        model.save(model_path)
    
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    print("\n💡 Escribe prompts en inglés (o prueba español simple)")
    print("📝 Ejemplos: 'Hello', 'The cat', 'Hola', 'El perro'")
    print("❌ Escribe 'quit' para salir\n")
    
    while True:
        try:
            prompt = input("> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            if not prompt:
                continue
            
            # Codificar y generar
            tokens = tokenizer.encode(prompt)
            print(f"Tokens: {tokens}")
            
            generated = model.generate(
                tokens, 
                max_tokens=15, 
                temperature=0.7,
                top_p=0.9
            )
            
            # Decodificar y mostrar
            text = tokenizer.decode(generated)
            print(f"📤: {text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Reiniciando...")
            model = MiniLLM(config)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mini-LLM para Chromebook')
    parser.add_argument('--mode', choices=['generate', 'interactive', 'benchmark'], 
                       default='interactive')
    parser.add_argument('--prompt', type=str, default="Hello")
    parser.add_argument('--model', type=str, default='minillm_model.pkl')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Configuración minimal
        config = Config(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=64
        )
        
        # Cargar o crear modelo
        if os.path.exists(args.model):
            model = MiniLLM.load(args.model)
        else:
            model = MiniLLM(config)
            model.save(args.model)
        
        tokenizer = SimpleTokenizer(config.vocab_size)
        
        # Generar
        print(f"📝 Prompt: {args.prompt}")
        tokens = tokenizer.encode(args.prompt)
        generated = model.generate(tokens, max_tokens=20, temperature=0.7)
        text = tokenizer.decode(generated)
        
        print(f"🤖 Generado: {text}")
        
    elif args.mode == 'interactive':
        interactive_demo()
        
    elif args.mode == 'benchmark':
        print("⏱️  Ejecutando benchmark...")
        
        config = Config(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=64
        )
        
        model = MiniLLM(config)
        tokenizer = SimpleTokenizer(config.vocab_size)
        
        # Benchmark de velocidad
        prompt = "The quick brown fox"
        tokens = tokenizer.encode(prompt)
        
        start = time.time()
        for _ in range(5):
            model.generate(tokens, max_tokens=10, temperature=0.0)
        end = time.time()
        
        avg_time = (end - start) / 5
        print(f"⏱️  Tiempo promedio por generación (10 tokens): {avg_time:.3f}s")
        print(f"⚡ Tokens por segundo: {10/avg_time:.1f}")
        
        # Uso de memoria aproximado
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"💾 Memoria usada: {mem_mb:.1f} MB")

if __name__ == "__main__":
    main()