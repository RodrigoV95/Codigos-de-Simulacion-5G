from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === PARÁMETROS FÍSICOS ===
f_mhz = 3500
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1.0                         # Distancia de referencia en metros
N = 2.5                          # Exponente de pérdida (outdoor)
Pt_dBm = 33                     # Potencia de transmisión

# === FSPL A 1 METRO (modelo CI) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c)

# === SMALL CELLS ===
small_cells = [
    (200, 300),  # techo abierto
    (450, 550),  # grada norte
    (400, 150)   # grada suroeste
]

# === CARGA DEL PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\EstadioPynandi.png'
img = Image.open(image_path).convert("L")
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === INICIALIZACIÓN DEL HEATMAP ===
heatmap = np.full((height, width), -150.0)

# === PROPAGACIÓN MODELO CLOSE-IN ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0  # evitar log(0) o distancias menores a d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === VISUALIZACIÓN ===
vmin, vmax = np.min(heatmap), np.max(heatmap)

plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin, vmax=vmax)

# === SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

plt.title("Simulación 5G Outdoor - Estadio Pynandi")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.show()
