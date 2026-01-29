from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

with open('mejor_modelo_exam.pkl', 'rb') as archivo:
    modelo_completo = pickle.load(archivo)

app = FastAPI(title="Predicción de Examen")

class DatosEstudiante(BaseModel):

    hours_spent_learning_per_week: float
    projects_completed: float
    debugging_sessions_per_week: float
    self_reported_confidence_python: float
    
    #ASI ESTABA ANTES:


    # hours_spent_learning_per_week: float
    # projects_completed: float
    # debugging_sessions_per_week: float
    # self_reported_confidence_python: float
    # attendance_rate: float
    # sleep_hours: float
    # previous_score: float
    # age: float
    # weeks_in_course: float
    # tutorial_videos_watched: float
    # practice_problems_solved: float
    # country: str
    # uses_kaggle: str
    # participates_in_discussion_forums: str

@app.post("/predecir")
async def predecir(estudiante: DatosEstudiante):
    df_entrada = pd.DataFrame([estudiante.dict()])
    
    prediccion = modelo_completo.predict(df_entrada)
    probabilidad = modelo_completo.predict_proba(df_entrada)[:, 1]
    
    return {
        "pasara_el_examen": int(prediccion[0]),
        "probabilidad_exito": float(probabilidad[0])
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Servidor funcionando correctamente"}