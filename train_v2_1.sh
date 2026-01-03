#!/bin/bash

# Script de entrenamiento para Vertical LLM
# Entrena en PyTorch y transfiere a NumPy

set -e

echo "=========================================="
echo "🧠 ENTRENAMIENTO VERTICAL LLM"
echo "=========================================="

# Configuración
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-0.002}
DEVICE=${DEVICE:-cpu}

# Verificar PyTorch
echo "[1/4] Verificando dependencias..."
python3 -c "import torch" 2>/dev/null || {
    echo "Instalando PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  --quiet
}

# Verificar NumPy
python3 -c "import numpy" 2>/dev/null || {
    echo "Instalando NumPy..."
    pip install numpy --quiet
}

echo "[2/4] Entrenamiento ligero para Chromebook..."
echo "   • Épocas: $EPOCHS"
echo "   • Batch size: $BATCH_SIZE"
echo "   • Learning rate: $LEARNING_RATE"
echo "   • Device: $DEVICE"

echo "[3/4] Ejecutando entrenamiento..."
echo "⚠️  Esto puede tomar varios minutos en Chromebook..."
echo "============================================================"

python3 train_pytorch_to_numpy.py \
    --mode train-light \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --model_pytorch "vertical_model_pytorch.pt" \
    --model_numpy "vertical_model_numpy.pkl"

echo "[4/4] Procesando resultados..."
echo "============================================================"

# Verificar archivos generados
if [ -f "vertical_model_numpy.pkl" ]; then
    echo "✅ Modelo NumPy generado: vertical_model_numpy.pkl"
    echo ""
    echo "🧪 Probando el modelo..."
    python3 train_pytorch_to_numpy.py --mode test --model_numpy "vertical_model_numpy.pkl"
else
    echo "❌ Error: No se generó el modelo"
    exit 1
fi

echo ""
echo "🎉 ¡Entrenamiento completado!"
echo ""
echo "📦 Archivos generados:"
echo "   • vertical_model_pytorch.pt  - Modelo PyTorch"
echo "   • vertical_model_numpy.pkl   - Modelo NumPy (para API)"
echo ""
echo "🚀 Para usar en la API:"
echo "   cp vertical_model_numpy.pkl vertical_model.pkl"
echo "   ./api.sh start"
echo ""
echo "🔧 Configuración del modelo entrenado:"
python3 -c "
import pickle
with open('vertical_model_numpy.pkl', 'rb') as f:
    data = pickle.load(f)
config = data['config']
print(f'   • Vocabulario: {config[\"vocab_size\"]}')
print(f'   • d_model: {config[\"d_model\']}')
print(f'   • Capas: {config[\"n_layers\"]}')
print(f'   • Embedding shape: {data[\"embedding\"].shape}')
"