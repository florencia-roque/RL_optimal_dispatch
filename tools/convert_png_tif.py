# tools/convert_png_tif.py

from PIL import Image

# Nombre de tu archivo exportado de draw.io
input_path = "tools/Diagrama.drawio.png" 
output_path = "tools/diagrama_final_cmes.tif"

try:
    # Abrir el PNG de alta resolución
    img = Image.open(input_path)
    
    # Forzar modo RGB (elimina cualquier transparencia residual)
    if img.mode != 'RGB':
        print(f"Convirtiendo de {img.mode} a RGB...")
        img = img.convert('RGB')
    
    # Guardar como TIFF cumpliendo normas CMES
    # DPI=600 para Combo, o 900 si es puro Line Art (texto y lineas negras)
    img.save(
        output_path, 
        dpi=(600, 600),             # 
        compression="tiff_lzw"      # Compresión sin pérdida
    )
    
    print(f"¡Listo! Guardado como {output_path}")
    print("Cumple con: TIFF, RGB, Sin transparencia, Alta Resolución.")

except Exception as e:
    print(f"Error: {e}")