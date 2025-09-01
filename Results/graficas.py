import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
BASE_DIR = r"D:\AutoAprendizaje\workspace_sionna\Results"
OUTPUT_DIR = os.path.join(BASE_DIR, "comparativas")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo bonito
plt.style.use("seaborn-v0_8-darkgrid")

# Colores consistentes
colors = {
    "neural_receiver1": "tab:blue",
    "neural_receiver2": "tab:orange",
    "neural_receiver3": "tab:green",
}

# === FUNCIÓN PARA PROCESAR UNA MODULACIÓN ===
def plot_modulation(modulation):
    modulation_dir = os.path.join(BASE_DIR, modulation)
    networks = ["neural_receiver1", "neural_receiver2", "neural_receiver3"]

    data_dict = {}

    for net in networks:
        # Buscar CSV dentro del último checkpoint
        net_dir = os.path.join(modulation_dir, net)
        if not os.path.exists(net_dir):
            continue

        # Buscar carpeta más reciente (ej. checkpoint_v69)
        checkpoints = [d for d in os.listdir(os.path.join(net_dir, "2025-09-01")) if d.startswith("checkpoint")]
        if not checkpoints:
            continue
        checkpoints.sort(reverse=True)
        latest_checkpoint = checkpoints[0]

        # Buscar CSV
        csv_path = os.path.join(net_dir, "2025-09-01", latest_checkpoint, f"{net}_{latest_checkpoint}_results.csv")
        if not os.path.isfile(csv_path):
            continue

        # Leer CSV
        df = pd.read_csv(csv_path)
        data_dict[net] = df

    # === Graficar BER ===
    plt.figure(figsize=(8, 6))
    for net, df in data_dict.items():
        plt.semilogy(df["EbNo_dB"], df["BER"], marker="o", label=net, color=colors[net])
    plt.xlabel("Eb/No [dB]")
    plt.ylabel("BER (log scale)")
    plt.title(f"BER vs Eb/No - {modulation.upper()}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{modulation}_BER.png"), dpi=300)
    plt.close()

    # === Graficar BLER ===
    plt.figure(figsize=(8, 6))
    for net, df in data_dict.items():
        plt.semilogy(df["EbNo_dB"], df["BLER"], marker="s", label=net, color=colors[net])
    plt.xlabel("Eb/No [dB]")
    plt.ylabel("BLER (log scale)")
    plt.title(f"BLER vs Eb/No - {modulation.upper()}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{modulation}_BLER.png"), dpi=300)
    plt.close()


# === MAIN ===
if __name__ == "__main__":
    modulations = ["qpsk", "qam16", "qam64", "qam256"]
    for mod in modulations:
        plot_modulation(mod)

    print(f"✅ Gráficos guardados en: {OUTPUT_DIR}")
