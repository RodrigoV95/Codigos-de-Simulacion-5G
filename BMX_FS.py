from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DE PARÁMETROS REALES OUTDOOR (ITU-R M.2412 CI) ===
f_mhz = 3500              # Frecuencia en MHz
f_hz = f_mhz * 1e6        # Frecuencia en Hz
Pt_dBm = 33               # Potencia de transmisión de small cells
N = 2.5                   # Exponente de pérdida para outdoor
c = 3e8                   # Velocidad de la luz en m/s
d0 = 1                    # Distancia de referencia en metros

# === CONSTANTE DEL MODELO CI ===
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === CARGAR PLANO ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\BMX_FS.png'
img = Image.open(image_path).convert('L')
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === POSICIONES DE SMALL CELLS ===
small_cells = [
    (400, 50),    # lado norte
    (400, 450)    # lado sur
]

# === SIMULACIÓN DE COBERTURA EN dBm ===
heatmap = np.full((height, width), -150.0, dtype=float)
for (cx, cy) in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1  # evita log(0)
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

# === DIBUJAR SMALL CELLS ===
for (cx, cy) in small_cells:
    plt.plot(cx, cy, 'wo', markersize=10, markeredgecolor='k', label='Small Cell (33 dBm)')

# === CONFIGURACIÓN DEL GRÁFICO ===
plt.title("Simulación 5G - BMX FS")
plt.xlabel("Pixels (X)")
plt.ylabel("Pixels (Y)")
plt.legend()

# === BARRA DE COLOR ===
cbar = plt.colorbar(img_plot)
cbar.set_label("Nivel de señal [dBm]")

# === EXPORTAR RESULTADO ===
plt.tight_layout()
plt.savefig("bmx_fs_simulacion_dBm_real_outdoor.png", dpi=300, bbox_inches="tight")
plt.show()
