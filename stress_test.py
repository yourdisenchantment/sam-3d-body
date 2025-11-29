import torch
import time
import sys

def stress_test(duration_sec=300):
    print(f"🔥 STARTING GPU STRESS TEST on {torch.cuda.get_device_name(0)}")
    print("Цель: Проверить перегрев памяти и стабильность питания.")
    
    # 1. Забиваем VRAM (Memory Stress)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    target_mem = int(total_mem * 0.90) # 90% памяти
    print(f"Allocating {target_mem / 1024**3:.2f} GB of VRAM...")
    
    try:
        # Создаем огромные тензоры
        size = target_mem // 4 // 2 # float32 = 4 bytes, делим на 2 тензора
        a = torch.randn(size, device='cuda')
        b = torch.randn(size, device='cuda')
        print("✅ VRAM заполнена.")
    except Exception as e:
        print(f"❌ Ошибка выделения памяти: {e}")
        return

    start_time = time.time()
    iter = 0
    
    # 2. Compute Stress (Матричные умножения греют чип)
    print("🚀 Running heavy compute loops...")
    
    try:
        while (time.time() - start_time) < duration_sec:
            # Тяжелая математика, чтобы нагрузить ядра и VRM
            c = torch.matmul(a[:10000], b[:10000]) 
            torch.cuda.synchronize()
            
            # Перегоняем данные туда-сюда (греет память)
            d = c.cpu()
            del c
            
            iter += 1
            if iter % 100 == 0:
                print(f"Iter {iter}: System alive. Elapsed: {time.time() - start_time:.1f}s")
                
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"\n❌ CRASH DETECTED: {e}")

    print("✅ Тест завершен без падения Python (но проверь системные логи).")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA!")
        sys.exit(1)
    stress_test()
