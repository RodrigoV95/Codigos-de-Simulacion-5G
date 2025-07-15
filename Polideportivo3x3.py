from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS ===
f_mhz = 3500
f_hz = f_mhz * 1e6
Pt_dBm = 33
N = 2.5
c = 3e8
d0 = 1

# === PÉRDIDA EN LA DISTANCIA DE REFERENCIA (Modelo CI - ITU-R M.2412) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === COORDENADAS DE LAS SMALL CELLS ===
small_cells = [
    (100, 150),
    (400, 150)
]

# === CARGA DEL PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\Polideportivo3x3.png'
img = Image.open(image_path).convert('L')
width, height = img.size

# === MALLA DE CÁLCULO ===
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0, dtype=float)

# === SIMULACIÓN DE SEÑAL CON MODELO CI ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === ESCALA AUTOMÁTICA EN dBm ===
vmin_dbm = np.min(heatmap)
vmax_dbm = np.max(heatmap)

# === VISUALIZACIÓN ===
plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin_dbm, vmax=vmax_dbm)

# === MARCAR POSICIÓN DE SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

plt.title("Simulación 5G - Polideportivo 3x3")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR REAL (en dBm) ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.savefig("polideportivo_3x3_simulacion_dBm_CI.png", dpi=300, bbox_inches="tight")
plt.show()
