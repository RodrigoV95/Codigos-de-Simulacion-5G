
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\CentroAcuatico.png'
f_mhz = 3500             # Frecuencia central en MHz (banda 5G)
N = 3.2                  # Exponente de pérdida
Pt_scs = 33              # Small Cells en dBm
Pt_rru = 33              # RRUs en dBm
radio_rru_px = 160       # Alcance aproximado de RRU en píxeles

# === CARGA DE IMAGEN ===
img = Image.open(image_path).convert("L")
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === ZONAS PROHIBIDAS (Piscinas delimitadas) ===
zonas_prohibidas = [
    ((140, 130), (665, 300)),   # piscinas
]

# === CÁLCULO DEL FSPL PARA DISTANCIA DE REFERENCIA (modelo CI indoor) ===
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1  # 1 metro
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis


# === FUNCIÓN DE VALIDACIÓN ===
def fuera_de_zonas_prohibidas(x, y):
    for (x0, y0), (x1, y1) in zonas_prohibidas:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
    return True

# === SMALL CELLS ===
small_cells = [(150, 50), (480, 50), (300, 350)]
small_cells = [sc for sc in small_cells if fuera_de_zonas_prohibidas(*sc)]

# === HEATMAP DE SMALL CELLS EN dBm ===
heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)        # Distancia a cada píxel
    d[d < 1] = 1                                    # Evita log(0)
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)        #Modelo de propagacion Close-in
    Pr = Pt_scs - PL                                # Potencia recibida en dBm
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === RRUs ORDENADAS + ALEATORIAS ===
rrus_candidatas = []
for x in range(60, width - 60, 50):
    for y in range(60, height - 60, 40):
        if fuera_de_zonas_prohibidas(x, y):
            rrus_candidatas.append((x, y))

rrus_iniciales = rrus_candidatas[:70]
faltantes = 70 - len(rrus_iniciales)

# Añadir RRUs aleatorias si no se llegó a 70
np.random.seed(99)
extras = []
while len(extras) < faltantes:
    x = np.random.randint(50, width - 50)
    y = np.random.randint(50, height - 50)
    if fuera_de_zonas_prohibidas(x, y):
        muy_cerca = any(np.hypot(x - rx, y - ry) < 35 for rx, ry in (rrus_iniciales + extras))
        if not muy_cerca:
            extras.append((x, y))

rrus = rrus_iniciales + extras

# === HEATMAP DE RRUs EN dBm ===
heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < 1] = 1
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === COMBINACIÓN DE AMBOS MAPAS EN dBm ===
combined_heatmap_dbm = 10 * np.log10(10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10))

# === VISUALIZACIÓN FINAL ===
vmin_dbm = np.min(combined_heatmap_dbm)
vmax_dbm = np.max(combined_heatmap_dbm)

plt.figure(figsize=(11, 7))
plt.imshow(img, extent=(0, width, 0, height), cmap="gray")
img_plot = plt.imshow(combined_heatmap_dbm, cmap="jet", alpha=0.65,
                      extent=(0, width, 0, height), origin="lower",
                      interpolation="bilinear", vmin=vmin_dbm, vmax=vmax_dbm)

# Dibujar Small Cells
plt.scatter(*zip(*small_cells), c='white', edgecolors='black',
            s=80, marker='o', label='Small Cells (33 dBm)')

# Dibujar RRUs
plt.scatter(*zip(*rrus), c='white', edgecolors='red',
            s=35, marker='s', label=f'RRUs (33 dBm) - {len(rrus)} nodos')

# Configuración visual del gráfico
plt.title("Simulación 5G - Centro Acuático")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')

# Barra de color
cbar = plt.colorbar(img_plot, shrink=0.8, pad=0.02)
cbar.set_label("Nivel de señal [dBm]")

# Guardar imagen final
plt.tight_layout()
plt.savefig("centro_acuatico_70_rrus_dBm.png", dpi=300, bbox_inches="tight")
plt.show()
