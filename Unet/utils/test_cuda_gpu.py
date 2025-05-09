import torch

# Vérifier si CUDA est disponible
if torch.cuda.is_available():
    print("CUDA is available.")
    # Obtenir le nombre de GPU disponibles
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Obtenir les propriétés de chaque GPU
    for i in range(num_gpus):
        print(f"GPU {i}:")
        # Obtenir le nom du GPU
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  Name: {gpu_name}")
        # Obtenir les propriétés détaillées du GPU
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"  Multi-Processor Count: {gpu_properties.multi_processor_count}")
        print(f"  Major Compute Capability: {gpu_properties.major}")
        print(f"  Minor Compute Capability: {gpu_properties.minor}")
else:
    print("CUDA is not available.")
