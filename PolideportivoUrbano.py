from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACION GENERAL ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\PolideportivoUrbano.png'  # Ruta del plano en formato PNG
f_mhz = 3500                # Frecuencia de operación en MHz
f_hz = f_mhz * 1e6          # Conversión a Hz
c = 3e8                     # Velocidad de la luz en m/s
d0 = 1                      # Distancia de referencia (1 metro)
N = 3.2                     # Exponente de pérdida de trayectoria
Pt_scs = 33                 # Potencia de transmisión de Small Cells (dBm)
Pt_rru = 33                 # RRUs en dBm

# === CÁLCULO DE LA PÉRDIDA A 1 METRO (FSPL_d0) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === CARGA DEL PLANO ===
img = Image.open(image_path).convert("L")          # Carga y convierte el plano a escala de grises
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === DEFINICION DE ZONA PROHIBIDA ===
zona_prohibida = [((200, 150), (440, 410))]  # Coordenadas aproximadas del área de competencia (tatamis)

# Verifica si una coordenada está fuera de la zona prohibida
def fuera_de_zona(x, y):
    for (x0, y0), (x1, y1) in zona_prohibida:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
    return True

# === SMALL CELLS ===
small_cells = [(100, 100), (550, 500)]  # Zona suroeste y noreste
small_cells = [sc for sc in small_cells if fuera_de_zona(*sc)]

# === HEATMAP DE SMALL CELLS EN dBm ===
heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_scs - PL
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === RRUs ===
rrus = [(180, 300), (480, 300), (320, 100), (320, 450)]
rrus = [r for r in rrus if fuera_de_zona(*r)]

# === HEATMAP DE RRUs EN dBm ===
heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === COMBINACION DE HEATMAPS EN dBm ===
combined_heatmap_dbm = 10 * np.log10(
    10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10)
)

# === VISUALIZACION ===
plt.figure(figsize=(10, 8))
plt.imshow(img, extent=(0, width, 0, height), cmap="gray")

# Mapa de calor en escala automática
img_plot = plt.imshow(combined_heatmap_dbm, cmap="jet", alpha=0.65,
                      extent=(0, width, 0, height), origin="lower",
                      interpolation="bilinear")

# Dibujar Small Cells y RRUs
plt.scatter(*zip(*small_cells), c='white', edgecolors='black', s=80, marker='o', label='Small Cells (33 dBm)')
plt.scatter(*zip(*rrus), c='white', edgecolors='red', s=40, marker='s', label=f'RRUs (33 dBm) - {len(rrus)} nodos')

plt.title("Simulación 5G - Polideportivo Urbano")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

# Barra de color real en dBm
cbar = plt.colorbar(img_plot, shrink=0.8, pad=0.02, aspect=30)
cbar.set_label("Nivel de señal [dBm]")

# === GUARDAR RESULTADO ===
plt.savefig("polideportivo_urbano_simulacion_dBm_real.png", dpi=300, bbox_inches="tight")
plt.show()
