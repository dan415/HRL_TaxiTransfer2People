

<h1>Proyecto de Aprendizaje por Refuerzo:</h1>

<h2>Método jerárquico de aprendizaje por refuerzo para el entorno Taxi-v3 de OpenAI Gym. </h2>


<h3>Introducción</h3>

Este proyecto se ha realizado como trabajo final de la asignatura de Aprendizaje por Refuerzo del Máster de Inteligencia Artificial de la Universidad Carlos III de Madrid.

<h4>Integrantes</h4>
Daniel Cabrera Rodríguez

...

...

<h3>Instalación</h3>

Usamos Python 3.10


Con conda:
Cambia el valor de "Prefix" en el fichero environment.yml a la ruta donde anaconda guarda los entornos virtuales.


En mi caso sería /home/dani/anaconda3/envs/refuerzo310

Normalmente, el entorno se creará en la ruta /home/{username}/anaconda3/envs/refuerzo310
```bash
conda create -f environment.yml
```


Con pip:


Aquí, conviene que primero crees un entorno virtual con venv

```bash
pip install -r requirements.txt
```
