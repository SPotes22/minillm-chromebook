#!/bin/bash

# Mini-LLM Runner - Versión 2.0 completamente funcional
# Optimizado para Chromebook

set -e  # Salir en error

echo "🚀 Mini-LLM para Chromebook"
echo "============================="
echo ""

# Configurar entorno para Chromebook
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Verificar Python
if ! command -v python3 &> /dev/null; then
    error "Python3 no encontrado. Por favor instálalo."
    exit 1
fi

# Verificar NumPy
info "Verificando dependencias..."
python3 -c "import numpy" 2>/dev/null || {
    warn "NumPy no encontrado. Instalando..."
    pip3 install numpy --user --quiet
    info "NumPy instalado ✓"
}

# Crear directorio para modelos si no existe
mkdir -p models

# Manejar argumentos
case "${1:-interactive}" in
    generate)
        PROMPT="${2:-"Hello world"}"
        info "Modo: Generación"
        info "Prompt: $PROMPT"
        python3 minillm_fixed_v2.py --mode generate --prompt "$PROMPT"
        ;;
        
    interactive)
        info "Modo: Interactivo"
        info "Presiona Ctrl+C para salir"
        echo ""
        python3 minillm_fixed_v2.py --mode interactive
        ;;
        
    benchmark)
        info "Modo: Benchmark"
        python3 minillm_fixed_v2.py --mode benchmark
        ;;
        
    train)
        warn "Modo entrenamiento no disponible en esta versión"
        info "Usa 'generate' o 'interactive' para probar el modelo"
        ;;
        
    clean)
        info "Limpiando archivos temporales..."
        rm -f minillm_model.pkl
        rm -f models/*.pkl
        info "Limpieza completada ✓"
        ;;
        
    help|--help|-h)
        echo "Uso: $0 {generate [prompt]|interactive|benchmark|train|clean|help}"
        echo ""
        echo "Comandos:"
        echo "  generate [prompt]  Generar texto con un prompt"
        echo "  interactive        Modo interactivo (predeterminado)"
        echo "  benchmark          Probar velocidad del modelo"
        echo "  train              (No implementado aún)"
        echo "  clean              Limpiar modelos guardados"
        echo "  help               Mostrar esta ayuda"
        echo ""
        echo "Ejemplos:"
        echo "  $0 generate \"Hello there\""
        echo "  $0 generate \"Hola mundo\""
        echo "  $0 interactive"
        echo "  $0 benchmark"
        ;;
        
    *)
        warn "Comando desconocido: $1"
        echo "Usa '$0 help' para ver opciones disponibles"
        echo ""
        info "Iniciando modo interactivo por defecto..."
        python3 minillm_fixed_v2.py --mode interactive
        ;;
esac

# Limpiar cache de Python
python3 -c "import sys; sys.version_info >= (3, 4) and __import__('importlib').invalidate_caches()" 2>/dev/null || true

echo ""
info "✅ Ejecución completada"