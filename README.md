# GDG_2026

Modelo entrenado para el reto GDG ML. Inferencia en CPU x86-64, con un hilo.

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Uso con el harness oficial

`model.py` está en la raíz y define `class Model` con `predict(sample) -> int`. El modelo carga sus archivos mediante rutas relativas a esta carpeta. No requiere entrenamiento ni acceso a internet durante la predicción.

Con el harness y los datos proporcionados por los organizadores:

```bash
python /ruta/al/harness.py evaluate --model ./model.py --source file:/ruta/a/test_demo.npz --runs 1
```

El preprocesado utiliza `preprocess_native.so` en Linux x86-64; si no puede cargarse, utiliza el respaldo NumPy incluido. Todos los archivos de esta carpeta son necesarios para ejecutar el modelo o documentar su uso.
