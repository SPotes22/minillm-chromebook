# Mini-LLM para Chromebook

## 🎯 Propósito
Implementación minimalista de un transformer optimizado para ejecutarse eficientemente en Chromebooks con recursos limitados.

## 🚀 Características principales

### ✅ Optimizaciones implementadas
- **Arquitectura vertical**: Código autocontenido sin dependencias pesadas
- **RoPE (Rotary Positional Encoding)**: Más eficiente que positional embeddings tradicionales
- **SwiGLU**: Activación más eficiente que GELU
- **KV-Cache**: Para generación rápida de tokens
- **Atención por chunks**: Reduce uso de memoria
- **LayerNorm optimizado**: Implementación minimalista

### 📊 Especificaciones técnicas
- **Modelo base**: Transformer decoder-only
- **Parámetros**: ~1-10M (ajustable)
- **Memoria**: < 512MB RAM
- **Dependencias**: Solo NumPy
- **Velocidad**: ~10-100 tokens/segundo en CPU

## 🛠️ Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/SPotes22/minillm-chromebook.git
cd minillm-chromebook

# 2. Instalar dependencias (solo NumPy)
chmod +x run.sh
./run.sh  # Esto verificará e instalará dependencias automáticamente
