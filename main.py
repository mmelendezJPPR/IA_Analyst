import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv #Lee el archivo .env
from openai import OpenAI # Importa la librería OpenAI


load_dotenv() #Carga las variables de entorno desde el archivo .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Inicializa el cliente OpenAI con la clave de API


def extraer_texto_con_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    texto = ""
    for pagina in doc:
        pix = pagina.get_pixmap(dpi=300)  # Más calidad
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        texto += pytesseract.image_to_string(img, lang="spa")  # idioma español
    return texto

# Dividir el texto si es muy largo
def dividir_texto(texto, tamaño=1500):
    return [texto[i:i + tamaño] for i in range(0, len(texto), tamaño)]

# Enviar fragmento a la IA
def analizar_fragmento(texto):
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",  # o "gpt-4" si tienes acceso
        messages=[
            {"role": "system", "content": "Resume el siguiente fragmento de un documento PDF:"},
            {"role": "user", "content": texto}
        ]
    )
    return respuesta.choices[0].message.content

# Ejecutar todo
if __name__ == "__main__":
    pdf_path = "ReglamentoEmergencia.pdf"
    texto = extraer_texto_con_ocr(pdf_path)

    if not texto.strip():
        print("⚠️ El PDF no tiene texto reconocible.")
    else:
        fragmentos = dividir_texto(texto)
        for i, f in enumerate(fragmentos):
            print(f"\n🔍 Fragmento {i+1}:\n{analizar_fragmento(f)}\n")
