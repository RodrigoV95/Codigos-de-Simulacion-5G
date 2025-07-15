from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GENERAL ===
image_path = r'C:\Users\Vera Asilvera\Pictures\Screenshots\Tirodeportivo.png'   # Ruta del plano
f_mhz = 3500                                # Frecuencia central 5G en MHz
N = 3.2                                     # Exponente de pérdida indoor
Pt_scs = 33                                 # Potencia de la small cell (dBm)
Pt_rru_dBm = 33                             # Potencia de las RRUs (dBm)
radio_rru_px = 160                          # Alcance estimado de cada RRU en píxeles

# === CARGA DEL PLANO ===
img = Image.open(image_path).convert("L")    # Plano en escala de grises
width, height = img.size
xx, yy = np.meshgrid(np.arange(width), np.arange(height))  # Malla de coordenadas

# === ZONA PROHIBIDA (rectángulo central) ===
zona_prohibida = [((53, 29), (441, 163))]   # Coordenadas exactas del área marcada

# === CÁLCULO DEL FSPL PARA d0 (modelo CI indoor) ===
f_hz = f_mhz * 1e6
c = 3e8
d0 = 1
FSPL_d0 = 20 * np.log10(4 * np.pi * d0 * f_hz / c) #Ecuacion de Friis

# Función para validar si una ubicación está fuera de la zona prohibida
def fuera_de_zona(x, y):
    for (x0, y0), (x1, y1) in zona_prohibida:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return False
    return True

# === UBICACIÓN DE LA SMALL CELL ===
small_cells = [(250, 200)]                   # Esquina izquierda
small_cells = [sc for sc in small_cells if fuera_de_zona(*sc)]

# === HEATMAP DE SMALL CELL EN dBm ===
heatmap_scs_dbm = np.full((height, width), -150.0)
for cx, cy in small_cells:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0  # Para evitar log(0)
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)    #Modelo de propagacion Close-in
    Pr = Pt_scs - PL
    heatmap_scs_dbm = np.maximum(heatmap_scs_dbm, Pr)

# === UBICACIÓN DE LAS 8 RRUs (todas arriba, donde estaban las X rojas) ===
rrus = [(x, 260) for x in np.linspace(90, 330, 8)]
rrus = [r for r in rrus if fuera_de_zona(*r)]

# === HEATMAP DE RRUs EN dBm ===
heatmap_rrus_dbm = np.full((height, width), -150.0)
for cx, cy in rrus:
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d[d < d0] = d0
    PL = FSPL_d0 + 10 * N * np.log10(d / d0)
    Pr = Pt_rru_dBm - PL
    heatmap_rrus_dbm = np.maximum(heatmap_rrus_dbm, Pr)

# === MAPA DE CALOR COMBINADO EN dBm ===
combined_heatmap_dbm = 10 * np.log10(10**(heatmap_scs_dbm / 10) + 10**(heatmap_rrus_dbm / 10))

# === ESCALA AUTOMÁTICA (valores reales en dBm) ===
vmin_dbm = np.min(combined_heatmap_dbm)
vmax_dbm = np.max(combined_heatmap_dbm)

# === VISUALIZACIÓN ===
plt.figure(figsize=(10, 6))
plt.imshow(img, extent=(0, width, 0, height), cmap="gray")  # Plano de fondo

# Mapa de calor superpuesto
img_plot = plt.imshow(combined_heatmap_dbm, cmap="jet", alpha=0.65,
                      extent=(0, width, 0, height), origin="lower",
                      interpolation="bilinear", vmin=vmin_dbm, vmax=vmax_dbm)

# Dibujar Small Cell y RRUs
plt.scatter(*zip(*small_cells), c='white', edgecolors='black', s=80, marker='o', label='Small Cell (33 dBm)')
plt.scatter(*zip(*rrus), c='white', edgecolors='red', s=40, marker='s', label='RRUs (33 dBm)')

# Detalles visuales
plt.title("Simulación 5G - Tiro Deportivo")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right')

# Barra de color (vertical estilo columna)
cbar = plt.colorbar(img_plot, shrink=0.8, pad=0.02, aspect=30)
cbar.set_label("Nivel de señal [dBm]")

# Guardar resultado
output_path = "tirodeportivo_simulacion_dBm_real.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
