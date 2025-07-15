from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS ===
f_mhz = 3500             # Frecuencia de la señal 5G    
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1                   # Distancia de referencia en metros
N = 2.5                  # Exponente de pérdida (outdoor)
Pt_dBm = 33              # Potencia de transmisión en dBm

# FSPL a 1 metro según modelo CI
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# Coordenadas de las small cells
small_cells = [
    (450, 450),           # lado sur
    (450, 50)             # lado norte
]

# === CARGA DEL PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\SkatePark.png'
img = Image.open(image_path).convert('L')
width, height = img.size

# === MALLA DE CÁLCULO ===
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0, dtype=float)

# === SIMULACIÓN DE SEÑAL (modelo CI en dBm) ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0  # evitar log(0) y mantener modelo CI válido
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === ESCALA AUTOMÁTICA ===
vmin_dbm = np.min(heatmap)
vmax_dbm = np.max(heatmap)

# === GRAFICAR MAPA DE CALOR SOBRE EL PLANO ===
plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin_dbm, vmax=vmax_dbm)

# === DIBUJAR SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

# === CONFIGURACIÓN DEL GRÁFICO ===
plt.title("Simulación 5G - Skate Park")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")

# === BARRA DE COLOR EN dBm ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.legend()
plt.tight_layout()
plt.savefig("skatepark_simulacion_CI_dBm.png", dpi=300, bbox_inches="tight")
plt.show()
