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
    env\Scripts\activate.bat
    ```

    - Para MacOs/Linux
    ```
    source env/bin/activate
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

    --data_folder_path: ruta a la carpeta donde guardamos los datos. Por default la ruta esta seteada a /data.

    --input_file_name: nombre del archivo con los datos de entrenamiento. Por default esta seteado a Train_BigMart.csv.

    --output_file_name: nombre del archivo para guardar la data transformada por el proceso de ingenieria de features. Por default esta seteado a train_data_transformed.csv.

    --model_file_name: nombre del archivo donde guardar el modelo entrenado. Por default esta setado a model.pickle.

2. Para correr el pipeline de inferencia, simplemente corremos el siguiente comando (siempre dentro de la carpeta del proyecto)
    ```
    python src/inference_pipeline.py
    ```

    Podemos incluir los siguientes argumentos opcionales:

    --data_folder_path: ruta a la carpeta donde guardamos los datos. Por default la ruta esta seteada a /data.
    
    --input_file_name: nombre del archivo con los datos de testeo. Por default esta seteado a Test_BigMart.csv.

    --transformed_data_file_name: nombre del archivo para guardar la data transformada por el proceso de ingenieria de features. Por default esta seteado a test_data_transformed.csv.

    --train_data_file_name: nombre del archivo con los datos de entrenamiento. Por default esta seteado a Train_BigMart.csv.

    --model_file_name: nombre del archivo que contiene el modelo entrenado. Por default esta setado a model.pickle.

    --output_file_name: nombre del archivo donde guardar las predicciones del modelo (dada la data de entrada). Por default esta seteado a predictions.csv.

    Por ejemplo, se puede probar el pipeline de inferencia sobre el archivo example.json corriendo el siguiente comando en consola:
    ```
    python src/inference_pipeline.py --input_file_name example.json
    ```
Tanto el pipeline de entrenamiento como de inferencia imprimen un log en la consola donde se puede ver si el algoritmo ejecuta todos los pasos correspondientes o arroja algun error.

### Ajuste de Hiper-parametros
Dentro de la carpeta /Notebook se encuentra el archivo hp_optimization.ipynb. 
En este notebook se realiza el ajuste de hiper-parametros para el modelo propuesto (regresion lineal). Se aplica el metodo Bayesiano de ajuste de hiper-parametros, utilizando la libreria *optuna*. Los parametros optimos encontrados son:

- *fit_intercept*: False
- *positive*: False
- *copy_X*: True
- *n_jobs*: 1