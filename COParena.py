from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\COParena.png'
f_mhz = 3500  # frecuencia en MHz
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1  # Distancia de referencia (1 metro)
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c)  # Pérdida en dB a 1 metro

N = 3.2  # exponente de perdida indoor
Pt_scs = 33  # potencia del small cell en dBm
Pt_rru = 33  # potencia del RRU en dBm

# === CARGAR PLANO ===
img = Image.open(image_path)
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === SMALL CELLS (irradiación real por modelo logarítmico CI) ===
small_cells = [
    (450, 500),  # Sala técnica
    (650, 320),  # Zona este
    (270, 140),  # Zona suroeste
]

heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr = Pt_scs - PL
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === RRUs (irradiación real con zonas rojas cubiertas) ===
rrus = [
    # Gradas superiores norte
    (160, 80), (280, 80), (400, 80), (520, 80), (605, 90),
    # Gradas inferiores sur
    (160, 580), (280, 580), (400, 580), (520, 580), (640, 580),
    # Laterales este
    (675, 165), (760, 240), (760, 330), (760, 420), (760, 500),
    # Laterales oeste
    (120, 140), (120, 240), (120, 330), (120, 420), (120, 500),
    # Pasillos intermedios
    (250, 250), (350, 250), (530, 250), (630, 250),
    (250, 420), (350, 420), (530, 420), (630, 420),
    # Vestuarios y técnica
    (300, 500), (580, 500), (440, 450), (440, 180),
    (200, 330), (680, 330)
]

heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === COMBINAR ambas coberturas en dBm ===
combined_heatmap_dbm = 10 * np.log10(10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10))

# === VISUALIZACIÓN ===
plt.figure(figsize=(12, 8))
plt.imshow(img, extent=(0, width, 0, height))

# Mapa de calor con escala automática real
img_map = plt.imshow(combined_heatmap_dbm, cmap='jet', alpha=0.65,
                     extent=(0, width, 0, height), origin='lower',
                     interpolation='bilinear')

# DIBUJAR NODOS
plt.scatter(*zip(*small_cells), c='white', edgecolors='black', s=60, marker='o', label='Small Cells (33 dBm)')
plt.scatter(*zip(*rrus), c='white', edgecolors='red', s=50, marker='s', label='RRUs (33 dBm)')

plt.title(" Simulación 5G - COP Arena ")
plt.xlabel("Coordenada X (px)", fontsize=12)
plt.ylabel("Coordenada Y (px)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

# === BARRA DE COLOR ===
cbar = plt.colorbar(img_map, shrink=0.8, pad=0.02, aspect=30)
cbar.set_label("Nivel de señal [dBm]")

# === EXPORTAR ===
plt.savefig("mapa_calor_SCs_RRUs_dBm_real.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("mapa_calor_SCs_RRUs_dBm_real.svg", format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

