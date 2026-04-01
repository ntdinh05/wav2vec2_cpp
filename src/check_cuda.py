import sys
import torch

def check_cuda():
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA built:  {torch.version.cuda}")
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available.")
        print("   Possible reasons:")
        print("   - No NVIDIA GPU detected")
        print("   - NVIDIA drivers not installed or outdated")
        print("   - PyTorch installed without CUDA support (e.g. CPU-only wheel)")
        print()
        print("   To install PyTorch with CUDA support, visit:")
        print("   https://pytorch.org/get-started/locally/")
        return

    print("✅ CUDA is available and ready!")
    print()
    print(f"  GPU count:      {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}:          {props.name}")
        print(f"    VRAM:         {total_mem_gb:.1f} GB")
        print(f"    CUDA arch:    sm_{props.major}{props.minor}")
        print(f"    Multiprocs:   {props.multi_processor_count}")
    print()

    # Quick functional test: run a small tensor op on GPU
    print("Running quick GPU tensor test...")
    try:
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print("✅ GPU tensor operation succeeded.")
    except Exception as e:
        print(f"❌ GPU tensor operation failed: {e}")
        return

    print()
    print("phoneme_benchmark.py will run on:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    check_cuda()
