from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS ===
f_mhz = 3500                     # Frecuencia central 5G (MHz)
f_hz = f_mhz * 1e6               # Frecuencia en Hz
c = 3e8                          # Velocidad de la luz (m/s)
N = 2.5                          # Exponente de pérdida (outdoor)
Pt_dBm = 33                      # Potencia de transmisión en dBm

# === CARGA DEL PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\Atletismo.png'
img = Image.open(image_path).convert('L')
width, height = img.size

# === COORDENADAS DE LAS SMALL CELLS (en píxeles) ===
small_cells = [
    (200, 50),
    (700, 50),
    (900, 500)
]

# === MALLA DE CÁLCULO ===
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0)

# === CÁLCULO DEL FSPL PARA DISTANCIA DE REFERENCIA (modelo CI) ===
d0 = 1  # 1 metro
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === SIMULACIÓN DE POTENCIA RECIBIDA EN dBm ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1  # Evitar log(0)
    PL = FSPL_d0 + 10 * N * np.log10(d / d0) #Modelo de propagacion Close-in
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

# === DIBUJAR SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

# === CONFIGURACIÓN GRÁFICA ===
plt.title("Simulación 5G - Atletismo")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR REAL ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")
cbar.set_ticks(np.linspace(vmin_dbm, vmax_dbm, 6))

plt.tight_layout()
plt.show()
