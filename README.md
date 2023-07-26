# AM2-TP-final
**Autor:  Federico Glancszpigel**

A continuacion se detalla una serie de pasos para poder correr el codigo dentro del 
repositorio. 

### Instalacion de dependencias
1. Clonar el repositorio en la consola (CMD en windows o terminal de Mac) con el siguiente comando

    ```
    git clone https://github.com/fedeglan/AM2-TP-final
    ```

    O tambien se puede clonar utilizando GitHub desktop.

2. Dentro del directorio del repositorio, crear un environment virtual con virtualenv.
En la consola (CMD en windows o terminal de Mac) correr el siguiente comando:

    ```
    python -m venv env
    ```

3. Activar el environment virtual con el siguiente comando

    - Para Windows:
    ```
    venv\Scripts\activate.bat
    ```

    - Para MacOs/Linux
    ```
    source venv/bin/activate
    ```

4. Ya con el environment virtual activado, procedemos a instalar los requerimientos.
En la consola correr el siguiente comando:
    ```
    pip install -r requirements.txt
    ```
    Recuerde siempre estar dentro de la carpeta del proyecto (AM2-TP-Final).

### Correr el *train_pipeline* y el *inference_pipeline*
1. Para entrenar el modelo simplemente corremos el siguiente comando (siempre 
dentro de la carpeta del proyecto)
    ```
    python src/train_pipeline.py
    ```

    Podemos incluir los siguientes argumentos opcionales:

    --input_path: ruta al archivo con los datos de entrenamiento. Por default la ruta esta seteada a /data/Train_BigMart.csv.

    --output_path: ruta al archivo donde guardar la data transformada por el proceso de ingenieria de features. Por default la ruta esta seteada a /data/train_data_transformed.csv.

    --model_path: ruta al archivo .pickle donde guardar el modelo entrenado. Por default la ruta esta setada a /data/model.pickle.

2. Para correr el pipeline de inferencia, simplemente corremos el siguiente comando (siempre dentro de la carpeta del proyecto)
    ```
    python src/inference_pipeline.py
    ```

    Podemos incluir los siguientes argumentos opcionales:

    --input_path: ruta al archivo con los datos de testeo. Por default la ruta esta seteada a /data/Test_BigMart.csv.

    --transformed_path: ruta al archivo donde guardar la data transformada por el proceso de ingenieria de features. Por default la ruta esta seteada a /data/test_data_transformed.csv.

    --model_path: ruta al archivo .pickle donde se encuentra el modelo entrenado. Por default la ruta esta setada a /data/model.pickle.

    --output_path: ruta al archivo donde guardar las predicciones del modelo (dada la data de entrada). Por default la ruta esta seteada a /data/predictions.csv.

Tanto el pipeline de entrenamiento como de inferencia imprimen un log en la consola donde se puede ver si el algoritmo ejecuta todos los pasos correspondientes o arroja algun error.

### Ajuste de Hiper-parametros
Dentro de la carpeta /Notebook se encuentra el archivo hp_optimization.ipynb. 
En este notebook se realiza el ajuste de hiper-parametros para el modelo propuesto (regresion lineal). Se aplica el metodo Bayesiano de ajuste de hiper-parametros, utilizando la libreria *optuna*. Los parametros optimos encontrados son:

- *fit_intercept*: False
- *positive*: False
- *copy_X*: True
- *n_jobs*: 1