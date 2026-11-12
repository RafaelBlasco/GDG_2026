# Modelo final: CNN destilada y cuantizada

Esta carpeta contiene el modelo completo para entregar al concurso. No necesita modelos anteriores, datos de entrenamiento ni acceso a Internet durante la inferencia.

## Archivos

- model.py: define Model() y predict(sample), que devuelve una clase 0–7 o -1 para abstención.
- model_int8.pt: red entrenada y exportada, con pesos y escalas de cuantización incluidos.
- preprocess.py: interpolación de valores ausentes y normalización por señal.
- config.json: nombre del grafo, backend, multiplicadores de clase y umbrales de abstención.
- requirements.txt: dependencias de ejecución.
- VALIDATION.json: comprobación independiente del paquete final.
- SHA256SUMS: integridad de los archivos entregados.

## Uso

Instalar las dependencias antes de evaluar:

```bash
python3 -m pip install -r requirements.txt
```

Desde el repositorio de materiales oficiales, evaluar esta carpeta con su harness:

```bash
python3 harness.py evaluate --model /ruta/entrega_final/model.py --source file:data/test_demo.npz --runs 1
```

El harness y los datos los proporcionan los organizadores; no son necesarios dentro de la entrega. Si se entrega mediante un repositorio GitHub, subir el contenido de esta carpeta para que model.py quede en la raíz del repositorio.

## Modelo y entorno

Una única CNN residual, destilada del ensemble original completo y cuantizada posteriormente. Usa solo las dos primeras de las cuatro señales de cada muestra. La inferencia se limita a un hilo de CPU. Todos los archivos se localizan respecto a model.py; no hay rutas al proyecto original.

Probado con Python 3.12, NumPy y PyTorch 2.14.1+cu130 en CPU x86 con AVX2, backend fbgemm. requirements.txt fija la versión pública de PyTorch probada. El grafo usa operadores cuantizados nativos y TorchScript; cambios de versión, plataforma o backend requieren una nueva comprobación de compatibilidad y exactitud. No se necesita GPU.

## Resultados y límites

Demo oficial de 400 ejemplos únicos: 357/400 correctos (89,25%), score 4,63194974; clase 6: 32/50. La comparación más reciente sobre demo repetida durante tres presupuestos completos de 60 segundos obtuvo score medio 58,49478 y latencia media 0,936 ms en esta máquina. El score extendido depende de la velocidad y de los ejemplos procesados; no es comparable directamente al score de 400 muestras.

Repetir la demo permite medir rendimiento sostenido, pero no añade ejemplos independientes. La demo y los conjuntos de validación se han observado en experimentos anteriores; estos resultados no garantizan generalización al test oculto. El modelo cuantizado conserva compromisos de exactitud frente al ensemble, especialmente en algunos casos de clase 6. No se modificaron parámetros al preparar esta entrega.
