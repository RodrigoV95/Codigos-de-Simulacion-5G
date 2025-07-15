from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
f_mhz = 3500                      # Frecuencia central 5G en MHz
f_hz = f_mhz * 1e6                # Frecuencia en Hz
Pt_dBm = 33                       # Potencia de transmisión en dBm (ej: macrocelda)
N = 2.5                           # Exponente de pérdida (outdoor)
c = 3e8                           # Velocidad de la luz
d0 = 1                            # Distancia de referencia (1 m)

# === CÁLCULO DE FSPL PARA d0 (ITU-R M.2412 - Modelo CI) ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === COORDENADAS DE SMALL CELLS ===
small_cells = [
    (100, 250),   # lado este
    (450, 250)    # lado oeste
]

# === CARGA DEL PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\EstadioHockey.png'
img = Image.open(image_path).convert('L')
width, height = img.size

# === MALLA DE CÁLCULO ===
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
heatmap = np.full((height, width), -150.0, dtype=float)

# === SIMULACIÓN DE POTENCIA RECIBIDA ===
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1  # para evitar log(0)
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr_dBm = Pt_dBm - PL
    heatmap = np.maximum(heatmap, Pr_dBm)

# === ESCALA AUTOMÁTICA ===
vmin_dbm = np.min(heatmap)
vmax_dbm = np.max(heatmap)

# === GRAFICAR MAPA DE CALOR ===
plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray', extent=(0, width, height, 0))
img_plot = plt.imshow(heatmap, cmap='jet', alpha=0.5,
                      extent=(0, width, height, 0),
                      vmin=vmin_dbm, vmax=vmax_dbm)

# === DIBUJAR POSICIONES DE SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell')

# === CONFIGURACIÓN FINAL ===
plt.title("Simulación 5G - Centro Nacional de Hockey")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR REAL EN dBm ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.savefig("hockey_simulacion_dBm_CI_auto.png", dpi=300, bbox_inches="tight")
plt.show()
