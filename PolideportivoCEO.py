from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\PolideportivoCEO.png'
f_mhz = 3500             # Frecuencia central en MHz (5G)
N = 3.2                  # Exponente de pérdida para indoor denso
Pt_scs = 33              # Potencia de small cells en dBm
Pt_rru = 33              # Potencia de RRUs en dBm
d0 = 1                   # Distancia de referencia (1 metro)

# === CÁLCULO DE FSPL EN d0 (modelo CI) ===
f_hz = f_mhz * 1e6
c = 3e8
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c)

# === CARGA DE IMAGEN ===
img = Image.open(image_path).convert("L")
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === ZONAS PROHIBIDAS (Tatamis y área azul) ===
zonas_prohibidas = [
    ((28, 27), (216, 258)),     # Dojo izquierda
    ((262, 51), (634, 244)),    # Área central
]

def fuera_de_zonas_prohibidas(x, y):
    for (x0, y0), (x1, y1) in zonas_prohibidas:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
    return True

# === SMALL CELLS ===
small_cells = [(100, 270), (450, 25), (670, 150)]
small_cells = [sc for sc in small_cells if fuera_de_zonas_prohibidas(*sc)]

# === HEATMAP DE SMALL CELLS EN dBm ===
heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr = Pt_scs - PL
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === RRUs CANDIDATAS ===
rrus_candidatas = []
for x in range(30, 700, 40):
    for y in [30, 65, 235, 270]:
        if fuera_de_zonas_prohibidas(x, y):
            rrus_candidatas.append((x, y))
for y in range(40, 300, 30):
    for x in [30, 260, 635, 700]:
        if fuera_de_zonas_prohibidas(x, y):
            rrus_candidatas.append((x, y))

# === FILTRADO DE RRUs ===
rrus = []
for rru in rrus_candidatas:
    if len(rrus) >= 35:
        break
    muy_cerca = any(np.hypot(rru[0] - rx, rru[1] - ry) < 25 for rx, ry in rrus)
    if not muy_cerca:
        rrus.append(rru)

# === HEATMAP DE RRUs EN dBm ===
heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === COMBINACIÓN DE MAPAS ===
combined_heatmap_dbm = 10 * np.log10(10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10))

# === ESCALA AUTOMÁTICA EN dBm ===
vmin_dbm = np.min(combined_heatmap_dbm)
vmax_dbm = np.max(combined_heatmap_dbm)

# === VISUALIZACIÓN ===
plt.figure(figsize=(11, 7))
plt.imshow(img, extent=(0, width, 0, height), cmap="gray")
img_plot = plt.imshow(combined_heatmap_dbm, cmap="jet", alpha=0.65,
                      extent=(0, width, 0, height), origin="lower",
                      interpolation="bilinear", vmin=vmin_dbm, vmax=vmax_dbm)

# Small Cells
plt.scatter(*zip(*small_cells), c='white', edgecolors='black',
            s=80, marker='o', label='Small Cells (33 dBm)')

# RRUs
plt.scatter(*zip(*rrus), c='white', edgecolors='red',
            s=35, marker='s', label=f'RRUs (33 dBm) - {len(rrus)} nodos')

plt.title("Simulación 5G - Polideportivo CEO")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')

# Barra de color en dBm
cbar = plt.colorbar(img_plot, shrink=0.8, pad=0.02)
cbar.set_label("Nivel de señal [dBm]")

# Guardar y mostrar
plt.tight_layout()
plt.savefig("polideportivo_CI.png", dpi=300, bbox_inches="tight")
plt.show()
