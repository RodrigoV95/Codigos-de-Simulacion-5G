from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS ===
f_mhz = 3500                     # Frecuencia en MHz
f_hz = f_mhz * 1e6               # Frecuencia en Hz
Pt_dBm = 33                      # Potencia de transmisión (small cells)
N = 2.5                          # Exponente de pérdida (outdoor realista)
c = 3e8                          # Velocidad de la luz (m/s)
d0 = 1                           # Distancia de referencia en metros

# === CÁLCULO DE PÉRDIDA EN d0 (ITU-R M.2412 - modelo CI) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === POSICIÓN DE SMALL CELLS ===
small_cells = [
    (350, 100),        # lado norte (bajo grada)
    (350, 450)         # lado sur (bajo grada)
]

# === CARGA DE PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\Patinodromo.png'
img = Image.open(image_path).convert('L')
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0)

# === SIMULACIÓN CON MODELO CI ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === ESCALA AUTOMÁTICA ===
vmin_dbm = np.min(heatmap)
vmax_dbm = np.max(heatmap)

# === VISUALIZACIÓN ===
plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin_dbm, vmax=vmax_dbm)

# === DIBUJAR SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

plt.title("Simulación 5G - Patinódromo")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR AUTOMÁTICA ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.savefig("patinodromo_simulacion_dBm_CI_auto.png", dpi=300, bbox_inches="tight")
plt.show()
