"""
Test rápido para verificar que el modelo UNet + Mamba funciona correctamente
"""

import torch
import torch.nn as nn
from model.unet_mamba_variants import UNetMamba

def test_model():
    print("🧪 TEST RÁPIDO: UNet + Mamba")
    print("-" * 40)
    
    # Verificar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")
    
    try:
        # Crear modelo
        print("🔧 Creando modelo...")
        model = UNetMamba(mode="full", strategy="integrate").to(device)
        print("✅ Modelo creado exitosamente")
        
        # Crear tensor de prueba
        print("🔧 Creando tensor de prueba...")
        test_input = torch.randn(1, 1, 256, 256).to(device)  # Batch=1, Channels=1, H=256, W=256
        print(f"✅ Input tensor: {test_input.shape}")
        
        # Forward pass
        print("🔧 Ejecutando forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Output tensor: {output.shape}")
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Verificar gradientes
        print("🔧 Verificando gradientes...")
        model.train()
        test_input.requires_grad_(True)
        output = model(test_input)
        loss = output.mean()
        loss.backward()
        
        print("✅ Gradientes calculados correctamente")
        print(f"✅ Loss: {loss.item():.4f}")
        
        print("\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("El modelo UNet + Mamba está funcionando correctamente.")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Listo para entrenar con el dataset completo!")
    else:
        print("\n❌ Hay problemas que necesitan resolverse.")
