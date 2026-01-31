# Modelo de predicción de Aprobación examen en Python

## Integrantes
* Daniela Casallas
* Diego Figueroa
* Hemersson Gutiérrez
* Jonathan Machuca

---

## Descripción del Problema
El desarrollo de este trabajo surge a partir de una colección de datos generados artificialmente y de manera netamente académica para responder a la pregunta de **si un estudiante de Python aprobará o no su examen**. Para esto, se manejan distintas variables asociadas al estudio y características del estudiante.

> **Nota:** El set de datos fue extraído desde Kaggle y el modelado del problema es de autoría del grupo (Publicado en Kaggle: [Link](https://www.kaggle.com/code/hemerssongutirrez/python-learning-multiple-models-using-pipeline )).

### Diccionario de Variables

| Columna | Tipo | Descripción |
| :--- | :--- | :--- |
| `student_id` | int | Identificador único para cada estudiante |
| `age` | int | Edad del estudiante (16–55) |
| `country` | object | País del estudiante (India, Bangladesh, EE.UU., Reino Unido, etc.) |
| `prior_programming_experience` | category | Nivel de experiencia previa en programación |
| `weeks_in_course` | int | Cantidad de semanas inscrito en el curso (1–15) |
| `hours_spent_learning_per_week` | float | Promedio de horas semanales dedicadas a Python |
| `practice_problems_solved` | int | Número de ejercicios de programación resueltos |
| `projects_completed` | int | Cantidad de proyectos en Python completados |
| `tutorial_videos_watched` | int | Número de videos relacionados con Python vistos |
| `uses_kaggle` | binary | Indica si el estudiante utiliza Kaggle (1 = sí) |
| `participates_in_discussion_forums` | binary | Indica si participa en foros de discusión |
| `debugging_sessions_per_week` | int | Frecuencia semanal de sesiones de depuración |
| `self_reported_confidence_python` | int | Nivel de confianza en Python (escala 1–10) |
| `final_exam_score` | float | Puntaje final del examen (0–100) |
| `passed_exam` | binary | Indicador de aprobación (1 = aprobado, 0 = reprobado) |

---

## Modelado
El problema es de **clasificación binaria**. Se implementaron todos los modelos aprendidos en el Diplomado, definiendo el **Accuracy** como métrica decisiva para la elección del mejor modelo. 

La justificación de la elección del Accuracy es netamente arbitraria dado el carácter académico del proyecto. Tras hacer competir los modelos mediante un **Pipeline**, se determinó que la **Regresión Logística con 4 variables** fue el modelo que maximizó el Accuracy de mejor forma.

---

## Instrucciones para correr la API localmente
1. **Abrir la carpeta del proyecto** en la terminal (utilizando `cd` o navegando hasta la ruta correspondiente).
2. **Ejecutar el servidor** con el siguiente comando:
   ```bash
   uvicorn M2_API:app --host 0.0.0.0 --port 8080 --reload

## Ejemplo de request al endpoint /predict:

{  
  &nbsp;"hours_spent_learning_per_week": 21,  
  &nbsp;"projects_completed": 6,  
  &nbsp;"debugging_sessions_per_week": 4,  
  &nbsp;"self_reported_confidence_python": 0.6  
}  

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| `hours_spent_learning_per_week` | 21 | Horas promedio por semana de aprendizaje en Python |
| `projects_completed` | 6 | Proyectos en Python completados en el curso |
| `debugging_sessions_per_week` | 4 | Promedio de sesiones haciendo debugging por semana |
| `self_reported_confidence_python` | 0.6 | Confianza autopercibida en programación en Python |

## Plataforma cloud usada para el deploy
Google Cloud Run
