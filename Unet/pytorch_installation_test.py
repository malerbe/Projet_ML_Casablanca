import torch

def main():
    # Vérifier si PyTorch est capable de voir le GPU
    print("### Vérification de la disponibilité du GPU ###")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device disponible : {device}")

    if device.type == "cuda":
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"Capacité de mémoire du GPU : {torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024):.2f} GB")
    else:
        print("Pas de GPU disponible. PyTorch utilisera le CPU.")

    # Tester une opération simple sur le device disponible
    print("\n### Test d'une opération simple sur le device disponible ###")
    try:
        x = torch.rand(5, 5).to(device)
        y = torch.rand(5, 5).to(device)
        z = x + y
        print(f"Résultat de l'addition de deux tenseurs sur {device} :\n{z}")
        print("PyTorch fonctionne correctement et utilise le", device)
    except Exception as e:
        print(f"Erreur lors de l'exécution d'une opération sur {device} : {e}")

if __name__ == "__main__":
    main()
