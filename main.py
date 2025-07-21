import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# ✅ CONFIGURACIÓN INICIAL
# =========================

# Ruta a Tesseract en tu sistema (ajustar si es necesario)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Cargar variables de entorno y la API de OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ✅ FUNCIONES AUXILIARES
# =========================

def extraer_tomo(pdf_path, desde_pagina, hasta_pagina):
    """Extrae un rango de páginas de un PDF original y lo guarda como tomo_extraido.pdf"""
    doc = fitz.open(pdf_path)
    sub_doc = fitz.open()
    sub_doc.insert_pdf(doc, from_page=desde_pagina, to_page=hasta_pagina)
    sub_doc.save("tomo_extraido.pdf")
    print("✅ Tomo extraído.")

def extraer_texto_con_ocr(pdf_path):
    """Realiza OCR a las primeras 20 páginas del PDF especificado"""
    doc = fitz.open(pdf_path)
    texto = ""
    for i, pagina in enumerate(doc):
        print(f"Procesando página {i+1}...")
        pix = pagina.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        texto += pytesseract.image_to_string(img, lang="spa")
        print("OCR terminado.")
    return texto

def dividir_texto(texto, tamaño=3500):
    """Divide el texto en fragmentos más pequeños para análisis"""
    return [texto[i:i + tamaño] for i in range(0, len(texto), tamaño)]

def analizar_fragmento(texto, instruccion):
    """Envía un fragmento de texto a la IA con una instrucción específica"""
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruccion},
            {"role": "user", "content": texto}
        ]
    )
    return respuesta.choices[0].message.content

# =========================
# ✅ PROMPTS PARA LA IA
# =========================

prompts_universal = {
    "prompt_1": "Establece una guía de referencia con flujograma a la luz del Reglamento Conjunto vigente sobre los trámites y requerimientos en el proceso de evaluación de transacciones de terrenos públicos por parte de la Junta de Planificación. Sé claro y organiza los pasos en forma lógica.",
    "prompt_2": "Establecer una guía de referencia con flujograma a la luz del Reglamento Conjunto vigente referente a los trámites y requerimientos para la evaluación de cambios de calificación directo por parte de la Junta de Planificación.",
    "prompt_3": "Establecer una guía de referencia con flujograma a la luz del Reglamento Conjunto vigente referente a los trámites y requerimientos para la evaluación de Sitios Históricos por parte de la Junta de Planificación.",
    "prompt_4": "Creación de una tabla donde se ilustren las columnas sobre cabida mínima y cabida máxima permitida para cada distrito de calificación.",
    "prompt_5": "Crear lista de Resoluciones de la JP por temas y año en que se suscribieron, para facilitar búsqueda de información.",
}

prompts_tomo_1 = {
    **prompts_universal,
    "prompt_6": ("6. Contestar las siguientes preguntas: Qué función tiene la División de Cumplimiento Ambiental? "
    "¿Cómo interaccionan otros Reglamentos con el Reglamento Conjunto?"),
}

prompts_tomo_2 = {
    **prompts_universal,
"prompt_6": (
    " Contestar las siguientes preguntas: ¿Cuáles son las disposiciones generales más importantes?\n"
    "2.    ¿Cuál es el alcance de este tomo?\n"
    "3.    Como se aplica la Ley 38-2017 LPAU\n"
    "4.    Resuma los servicios y tramites\n"
    "5.    Qué clases de solicitudes hay\n"
    "6.    Favor distinguir entre procesos adjudicativos y procesos cuasi-legislativos\n"
    "7.    Cuando se requieren vistas públicas\n"
    "8.    Qué es un asunto ministerial, y como se tramita\n"
    "9.    Cuantas clases de notificaciones hay\n"
    "10.   Solicitudes de carácter discrecional, que es, y como se tramitan\n"
    "11.   ¿Cuántos elementos y requisitos para aprobar una determinación final hay?\n"
),
}

prompts_tomo_3 = {
    **prompts_universal,
   "prompt_6": (
    """¿Qué es un permiso para desarrollo y negocios?
    ¿Qué requisitos tiene?
    ¿Cómo se tramita?
    ¿Cuándo una determinación final es un proceso adjudicativo, según la Ley 38-2017 LPAU?
    ¿Cuántos permisos hay en este tomo? Señalar diferencias en requisitos.
    ¿Qué es un permiso de medio ambiente? ¿Cuántos hay? Requisitos y cómo se tramitan."""
)

}
prompts_tomo_4 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Cuántas licencias y certificaciones hay?\n"
        "2. ¿Qué requisitos tienen?\n"
        "3. ¿Cómo se tramitan?\n"
        "4. ¿Qué negocios y operaciones requieren estas licencias y certificaciones?"
    )
}

prompts_tomo_5 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Qué es un proyecto de urbanización?\n"
        "2. ¿Qué es un proyecto de lotificación?\n"
        "3. ¿Cómo se utilizan las clasificaciones en estos trámites?\n"
        "4. ¿Cómo se utilizan las calificaciones?\n"
        "5. ¿Cuántas agencias y trámites se requieren para estos proyectos?"
    )
}

prompts_tomo_6 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Qué es una equivalencia?\n"
        "2. ¿Cuántas clasificaciones hay?\n"
        "3. ¿Cuántas calificaciones son similares y se pueden consolidar?\n"
        "4. ¿Cuántos planes especiales hay y qué requisitos tienen?\n"
        "5. ¿Cuántas calificaciones de conservación hay?\n"
        "6. Correlacionar las calificaciones de conservación y consolidar en menos.\n"
        "7. ¿Qué es un parámetro de diseño, cómo se utilizan?\n"
        "8. ¿Cuántas prohibiciones tienen las calificaciones?\n"
        "9. ¿Cómo interactúa el plan de uso de terrenos?\n"
        "10. ¿Cuántos procesos ambientales se utilizan en las calificaciones?\n"
        "11. ¿Qué certificaciones de agencia se requieren para las calificaciones?"
    )
}

prompts_tomo_7 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Cuántos procesos tiene la Junta de Planificación?\n"
        "2. ¿Qué requisitos tienen estos trámites?\n"
        "3. ¿Cómo se debe delimitar la Zona Costanera?"
    )
}

prompts_tomo_8 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Qué está sujeto a los parámetros de edificabilidad?\n"
        "2. ¿Cuántos parámetros de edificabilidad hay?\n"
        "3. ¿Cómo aplican estos parámetros en los distritos de calificación?\n"
        "4. Otros trámites en este tomo"
    )
}

prompts_tomo_9 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Qué son obras de infraestructura?\n"
        "2. ¿Cuántos trámites de infraestructura hay?\n"
        "3. Describir energía, acueductos y alcantarillados.\n"
        "4. ¿Qué recomendaciones se requieren en todos los procesos de este tomo?\n"
        "5. ¿Plan vial y acceso a vías, cómo se aprueban otros trámites en este tomo?"
    )
}

prompts_tomo_10 = {
    **prompts_universal,
    "prompt_6": (
        "Contestar las siguientes preguntas:\n"
        "1. ¿Qué es una zona o sitio histórico?\n"
        "2. ¿Cómo se nomina y se declara?\n"
        "3. ¿Qué restricciones presentan estos distritos?"
    )
}

prompts_tomo_11 = {
    **prompts_universal,
    "prompt_6": (
        "Resume el contenido del Tomo 11\n"
        "1. Resumir detalladamente"
    )
}

# =========================
# ✅ PROCESO COMPLETO POR TOMO
# =========================

def procesar_tomo_y_guardar_archivos(nombre_tomo , prompts):
    """Extrae texto con OCR y genera los archivos de flujogramas y tablas para un tomo"""
    texto = extraer_texto_con_ocr("tomo_extraido.pdf")

    if not texto.strip():
        print("⚠️ El PDF no tiene texto reconocible.")
        return

    with open(f"texto_extraido_{nombre_tomo}.txt", "w", encoding="utf-8") as f:
        f.write(texto)

    fragmentos = dividir_texto(texto)

    flujograma_1 = flujograma_2 = flujograma_3 = flujograma_4 = flujograma_5 = flujograma_6 = ""
    procesado_prompt_6 = False  # ← ESTA ES LA LÍNEA QUE TE FALTABA

    for i, fragmento in enumerate(fragmentos):
        resultado_1 = analizar_fragmento(fragmento, prompts["prompt_1"])
        resultado_2 = analizar_fragmento(fragmento, prompts["prompt_2"])
        resultado_3 = analizar_fragmento(fragmento, prompts["prompt_3"])
        resultado_4 = analizar_fragmento(fragmento, prompts["prompt_4"])
        resultado_5 = analizar_fragmento(fragmento, prompts["prompt_5"])

        flujograma_1 += f"\n🔍 Fragmento {i+1}:\n{resultado_1}\n"
        flujograma_2 += f"\n🔍 Fragmento {i+1}:\n{resultado_2}\n"
        flujograma_3 += f"\n🔍 Fragmento {i+1}:\n{resultado_3}\n"
        flujograma_4 += f"\n🔍 Fragmento {i+1}:\n{resultado_4}\n"
        flujograma_5 += f"\n🔍 Fragmento {i+1}:\n{resultado_5}\n"

        # Esta sección es la corrección clave ✅
        if not procesado_prompt_6:
            resultado_6 = analizar_fragmento(fragmento, prompts["prompt_6"])
            flujograma_6 += f"\n🔍 Fragmento {i+1}:\n{resultado_6}\n"
            procesado_prompt_6 = True
            break

       
    # Guardar resultados
    with open(f"flujogramaTerrPublicos_{nombre_tomo}.txt", "w", encoding="utf-8") as out1:
        out1.write(flujograma_1)
    with open(f"flujogramaCambiosCalificacion_{nombre_tomo}.txt", "w", encoding="utf-8") as out2:
        out2.write(flujograma_2)
    with open(f"flujogramaSitiosHistoricos_{nombre_tomo}.txt", "w", encoding="utf-8") as out3:
        out3.write(flujograma_3)
    with open(f"TablaCabida_{nombre_tomo}.txt", "w", encoding="utf-8") as out4:
        out4.write(flujograma_4)
    with open(f"Resoluciones_{nombre_tomo}.txt", "w", encoding="utf-8") as out5:
        out5.write(flujograma_5)
    with open(f"Respuestas_{nombre_tomo}.txt", "w", encoding="utf-8") as f6:
        f6.write(flujograma_6)

    print(f"✅ Análisis de {nombre_tomo} completado.\n")

# =========================
# ✅ FLUJO PRINCIPAL
# =========================

if __name__ == "__main__":
    # Procesar Tomo 1
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 52, 67)
    #procesar_tomo_y_guardar_archivos("Tomo_1" , prompts_tomo_1)

    # Procesar Tomo 2
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 68, 171)
    #procesar_tomo_y_guardar_archivos("Tomo_2" , prompts_tomo_2) 

    # Procesar Tomo 3
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 172, 257)
   #procesar_tomo_y_guardar_archivos("Tomo_3" , prompts_tomo_3)

    # Procesar Tomo 4
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 258, 319)
    #procesar_tomo_y_guardar_archivos("Tomo_4" , prompts_tomo_4)

    # Procesar Tomo 5
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 320, 357)
    #procesar_tomo_y_guardar_archivos("Tomo_5" , prompts_tomo_5)

    # Procesar Tomo 6
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 358, 538)
    #procesar_tomo_y_guardar_archivos("Tomo_6" , prompts_tomo_6)

    # Procesar Tomo 7
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 540, 587)
    #procesar_tomo_y_guardar_archivos("Tomo_7" , prompts_tomo_7)

    # Procesar Tomo 8
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 588, 657)
    #procesar_tomo_y_guardar_archivos("Tomo_8" , prompts_tomo_8)

    # Procesar Tomo 9
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 658, 736)
    #procesar_tomo_y_guardar_archivos("Tomo_9" , prompts_tomo_9)

    # Procesar Tomo 10
    #extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 737, 783)
    #procesar_tomo_y_guardar_archivos("Tomo_10" , prompts_tomo_10)

    # Procesar Tomo 11
    extraer_tomo("REGLAMENTO_CONJUNTO_2020.pdf", 784, 814)
    procesar_tomo_y_guardar_archivos("Tomo_11" , prompts_tomo_11)
