#!/usr/bin/env python3
"""
Script para monitorear el uso de GPU durante el entrenamiento
"""

import torch
import time
import psutil
import os

def print_system_info():
    """Imprime información del sistema"""
    print("=" * 60)
    print("🖥️  INFORMACIÓN DEL SISTEMA")
    print("=" * 60)
    
    # CPU
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # GPU
    if torch.cuda.is_available():
        print(f"CUDA disponible: ✅")
        print(f"Versión CUDA: {torch.version.cuda}")
        print(f"GPU(s) detectada(s): {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memoria total: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print(f"CUDA disponible: ❌")
    
    print("=" * 60)

def monitor_gpu_usage():
    """Monitorea el uso de GPU en tiempo real"""
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return
    
    print("🔍 Monitoreando uso de GPU (Ctrl+C para salir)...")
    print("Time\t\tGPU Usage\tMemory Used\tMemory Total")
    print("-" * 60)
    
    try:
        while True:
            # Obtener estadísticas de GPU
            gpu_usage = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_percent = (memory_used / memory_total) * 100
            
            current_time = time.strftime("%H:%M:%S")
            print(f"{current_time}\t{gpu_usage:>3}%\t\t{memory_used:>6.2f} GB\t{memory_total:>6.1f} GB ({memory_percent:>5.1f}%)")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoreo detenido")

def test_gpu_performance():
    """Test básico de rendimiento de GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible para test")
        return
    
    print("🧪 Ejecutando test de rendimiento GPU...")
    device = torch.device("cuda")
    
    # Test de operaciones básicas
    size = 5000
    print(f"Creando tensores de {size}x{size}...")
    
    start_time = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    print("Ejecutando multiplicación de matrices...")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Esperar a que termine
    
    end_time = time.time()
    print(f"✅ Test completado en {end_time - start_time:.2f} segundos")
    
    # Limpiar memoria
    del a, b, c
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "info":
            print_system_info()
        elif command == "monitor":
            print_system_info()
            monitor_gpu_usage()
        elif command == "test":
            print_system_info()
            test_gpu_performance()
        else:
            print("Comandos disponibles: info, monitor, test")
    else:
        print_system_info()
        print("\nComandos disponibles:")
        print("  python gpu_monitor.py info     - Mostrar información del sistema")
        print("  python gpu_monitor.py monitor  - Monitorear uso de GPU en tiempo real")
        print("  python gpu_monitor.py test     - Test de rendimiento básico")
