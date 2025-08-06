#!/usr/bin/env python3
"""
Script para cambiar entre modo TESTING y PRODUCCIÓN
"""
import os
import sys

def set_testing_mode():
    """Configura el entorno para modo TESTING/DESARROLLO"""
    os.environ['IS_TESTING'] = 'True'
    print("🧪 Modo TESTING activado")
    print("   - Entrenamiento rápido con pocas muestras")
    print("   - Optimizado para hardware limitado")
    print("   - Logging detallado para desarrollo")

def set_production_mode():
    """Configura el entorno para modo PRODUCCIÓN"""
    os.environ['IS_TESTING'] = 'False'
    print("🚀 Modo PRODUCCIÓN activado")
    print("   - Entrenamiento completo")
    print("   - Configuración para hardware potente")
    print("   - Logging optimizado para rendimiento")

def show_current_mode():
    """Muestra el modo actual"""
    is_testing = os.getenv('IS_TESTING', 'True').lower() == 'true'
    if is_testing:
        print("📊 Modo actual: TESTING/DESARROLLO")
    else:
        print("📊 Modo actual: PRODUCCIÓN")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python set_mode.py [testing|production|status]")
        print("  testing    - Activa modo testing/desarrollo")
        print("  production - Activa modo producción")
        print("  status     - Muestra el modo actual")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "testing":
        set_testing_mode()
    elif mode == "production":
        set_production_mode()
    elif mode == "status":
        show_current_mode()
    else:
        print(f"❌ Modo '{mode}' no reconocido")
        print("Modos disponibles: testing, production, status")
        sys.exit(1)
