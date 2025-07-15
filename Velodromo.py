from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
f_mhz = 3500              # Frecuencia central en MHz (banda 5G)
N = 3.2                   # Exponente de pérdida (escenario indoor)
Pt_scs = 33               # Potencia de small cell en dBm
Pt_rru = 33               # Potencia de RRU en dBm
radio_rru_px = 160        # Radio de cobertura estimado para cada RRU

# === CARGA DEL PLANO DEL VELODROMO ===
image_path = r'C:\Users\Vera Asilvera\Downloads\Velodromo.png'
img = Image.open(image_path).convert("L")
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))

# === DEFINICIÓN DE ZONA PROHIBIDA ===
zona_prohibida_velodromo = [((60, 50), (550, 350))]

def fuera_de_zona_velodromo(x, y):
    for (x0, y0), (x1, y1) in zona_prohibida_velodromo:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
    return True

# === CÁLCULO DEL FSPL PARA d0 = 1m (modelo CI indoor) ===
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) # Ecuacion de Friis

# === SMALL CELLS ===
small_cells_velodromo = [(300, 25), (300, 370)]
small_cells_velodromo = [sc for sc in small_cells_velodromo if fuera_de_zona_velodromo(*sc)]

# === HEATMAP DE SMALL CELLS EN dBm ===
heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells_velodromo:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr = Pt_scs - PL
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === UBICACIÓN DE 50 RRUs ===
rrus_alrededor = []
x_coords = np.linspace(60, 550, 20)
for x in x_coords:
    rrus_alrededor.append((int(x), 350))
    rrus_alrededor.append((int(x), 50))
y_coords_lat = np.linspace(50, 350, 5)
for y in y_coords_lat:
    rrus_alrededor.append((60, int(y)))
    rrus_alrededor.append((550, int(y)))
rrus_finales = rrus_alrededor[:50]

# === HEATMAP DE RRUs EN dBm ===
heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus_finales:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === COMBINACIÓN EN dBm ===
combined_heatmap_dbm = 10 * np.log10(10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10))

# === VISUALIZACIÓN FINAL EN dBm ===
vmin_dbm = np.min(combined_heatmap_dbm)
vmax_dbm = np.max(combined_heatmap_dbm)

plt.figure(figsize=(11, 7))
plt.imshow(img, extent=(0, width, 0, height), cmap="gray")
img_plot = plt.imshow(combined_heatmap_dbm, cmap="jet", alpha=0.65,
                      extent=(0, width, 0, height), origin="lower",
                      interpolation="bilinear", vmin=vmin_dbm, vmax=vmax_dbm)

# Dibujar Small Cells
plt.scatter(*zip(*small_cells_velodromo), c='white', edgecolors='black',
            s=80, marker='o', label='Small Cells (33 dBm)')

# Dibujar RRUs
plt.scatter(*zip(*rrus_finales), c='white', edgecolors='red',
            s=35, marker='s', label=f'RRUs (33 dBm) - {len(rrus_finales)} nodos')

plt.title("Simulación 5G - Velódromo")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')

# Barra de color en escala real
cbar = plt.colorbar(img_plot, shrink=0.8, pad=0.02)
cbar.set_label("Nivel de señal [dBm]")

plt.tight_layout()
plt.savefig("velodromo_simulacion_dBm_real.png", dpi=300, bbox_inches="tight")
plt.show()
