from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS ===
f_mhz = 3500             # Frecuencia en MHz
f_hz = f_mhz * 1e6       # Conversión a Hz
c = 3e8                  # Velocidad de la luz (m/s)
d0 = 1.0                 # Distancia de referencia (1 metro)
N = 2.5                  # Exponente de pérdida (outdoor)
Pt_dBm = 33              # Potencia de transmisión en dBm

# === FSPL A 1 METRO (Modelo Close-In) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c)

# === COORDENADAS DE SMALL CELLS ===
small_cells = [
    (360, 50),
    (500, 360)
]

# === CARGAR PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\BMXRace.png'
img = Image.open(image_path).convert('L')
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0)

# === SIMULACIÓN DE COBERTURA CON MODELO CI ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0  # Evitar log(0)
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === VISUALIZACIÓN EN ESCALA REAL ===
vmin, vmax = np.min(heatmap), np.max(heatmap)

plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin, vmax=vmax)

# === DIBUJAR SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

plt.title("Simulación 5G - BMX Race")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR REAL ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.show()
