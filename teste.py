import numpy as np
import matplotlib.pyplot as plt
import rasterio

# ======================================================
# CAMINHO DAS IMAGENS
# ======================================================

imagem_2023 = "landsat_2023.tif"
imagem_2024 = "landsat_2024.tif"

# ======================================================
# LEITURA DA BANDA NIR
# ======================================================

print("Lendo imagens...")

with rasterio.open(imagem_2023) as src2023:
    nir_2023 = src2023.read(5).astype(np.float32)

    perfil = src2023.profile

with rasterio.open(imagem_2024) as src2024:
    nir_2024 = src2024.read(5).astype(np.float32)

# ======================================================
# DIFERENÇA TEMPORAL
# ======================================================

print("Calculando diferença...")

diff = nir_2024 - nir_2023

# ======================================================
# LIMIAR
# ======================================================

threshold = 2 * np.std(diff)

# ======================================================
# MÁSCARA
# ======================================================

change_mask = np.abs(diff) > threshold

# ======================================================
# RESULTADOS
# ======================================================

print("\n===== RESULTADOS =====")

print(f"Mínimo: {diff.min():.2f}")
print(f"Máximo: {diff.max():.2f}")

print(f"Threshold: {threshold:.2f}")

print(f"Pixels alterados: {change_mask.sum():,}")

print(f"Percentual: {100 * change_mask.mean():.2f}%")

# ======================================================
# SALVAR GEOTIFF
# ======================================================

print("\nSalvando raster...")

perfil.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw"
)

with rasterio.open(
    "mascara_mudanca.tif",
    "w",
    **perfil
) as dst:

    dst.write(
        change_mask.astype(rasterio.uint8),
        1
    )

print("Raster salvo!")

# ======================================================
# VISUALIZAÇÃO
# ======================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ----------------------------------------
# DIFERENÇA
# ----------------------------------------

img = axes[0].imshow(
    diff,
    cmap="RdBu",
    vmin=-threshold * 2,
    vmax=threshold * 2
)

axes[0].set_title("Diferença NIR")
axes[0].axis("off")

plt.colorbar(img, ax=axes[0])

# ----------------------------------------
# HISTOGRAMA
# ----------------------------------------

axes[1].hist(
    diff.ravel(),
    bins=100,
    color="steelblue"
)

axes[1].axvline(
    threshold,
    color="red",
    linestyle="--"
)

axes[1].axvline(
    -threshold,
    color="red",
    linestyle="--"
)

axes[1].set_title("Histograma")

# ----------------------------------------
# MÁSCARA
# ----------------------------------------

axes[2].imshow(
    change_mask,
    cmap="Reds"
)

axes[2].set_title("Máscara de Mudança")

axes[2].axis("off")

plt.tight_layout()

plt.show()

print("\nProcessamento concluído!")