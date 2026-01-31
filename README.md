# Tarea_Cloud_Computing_2026

o Integrantes:
	- Daniela Casallas
	- Diego Figueroa
	- Hemersson Gutiérrez
	- Jonathan Machuca

o Descripción del problema:

El desarrollo de este trabajo es a partir de una colección de datos generados artificialmente y de manera netamente académica para responder la pregunta de si un estudiante de python aprobará o no su examen, para esto se manejan distintas variables asociadas al estudio y características del estudiante

El set de datos fue extraído desde Kaggle y el modelado del problema es de autoría del grupo (Publicado en Kaggle)

Las variables se describen a continuación:

+--------------------------------------+----------+-----------------------------------------------------------------------+
| Column                               | Type     | Description                                                           |
+--------------------------------------+----------+-----------------------------------------------------------------------+
| student_id                           | int      | Identificador único para cada estudiante                              |
| age                                  | int      | Edad del estudiante (16–55)                                           |
| country                              | object   | País del estudiante (India, Bangladesh, EE.UU., Reino Unido, etc.)    |
| prior_programming_experience         | category | Nivel de experiencia en programación antes de iniciar Python          |
| weeks_in_course                      | int      | Cantidad de semanas inscrito en el curso (1–15)                       |
| hours_spent_learning_per_week        | float    | Promedio de horas semanales dedicadas al aprendizaje de Python        |
| practice_problems_solved             | int      | Número de ejercicios de programación resueltos                        |
| projects_completed                   | int      | Cantidad de proyectos en Python completados                           |
| tutorial_videos_watched              | int      | Número de videos relacionados con Python vistos                       |
| uses_kaggle                          | binary   | Indica si el estudiante utiliza Kaggle (1 = sí)                       |
| participates_in_discussion_forums    | binary   | Indica si el estudiante participa en foros de discusión               |
| debugging_sessions_per_week          | int      | Frecuencia semanal de sesiones de depuración                           |
| self_reported_confidence_python      | int      | Nivel de confianza en Python (escala 1–10)                            |
| final_exam_score                     | float    | Puntaje final del examen (0–100)                                      |
| passed_exam                          | binary   | Indicador de aprobación del examen (1 = aprobado, 0 = reprobado)      |
+--------------------------------------+----------+-----------------------------------------------------------------------+

Modelado:


El problema es de clasificación, específicamente Binaria, dado que se quiere predecir si un estudiante aprobará o no su examen, para lo anterior se implementaron todos los modelos aprendidos en el Diplomado y se definió el Accuracy como metrica decisiva para la elección de cual modelo es el mejor, la justificación de la elección del Accuracy  es netamente arbitraria dado que este es un modelo académico y no real, posteriormente mediante Pipeline los modelos "compitieron" y se determinó que la Regresión Logistica con 4 variables era el modelo que maximizaba el Accuracy de mejor forma



o Instrucciones para correr la API localmente:

 - Abrir carpeta del proyecto en la terminal (utilizando CD, espacio y arrastrar carpeta o en su defecto navegar hasta la carpeta con comandos)
 - Ejecutar uvicorn M2_API:app --host 0.0.0.0 --port 8080 --reload

o Ejemplo de request al endpoint /predict:

{
  "hours_spent_learning_per_week": 21, (Horas promedio por semana de aprendizaje en python)
  "projects_completed": 6, (Proyectos en python completado en el curso)
  "debugging_sessions_per_week": 4, (Promedio de sesiones haciendo debugging por semana)
  "self_reported_confidence_python": 1 (Confianza auto percibida en programación en python)
}

o Plataforma cloud usada para el deploy: Google Cloud Run
