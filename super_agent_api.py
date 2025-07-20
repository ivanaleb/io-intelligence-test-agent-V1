
# Instalar dependencias:
# pip install fastapi uvicorn iointel openai python-multipart python-dotenv

import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import json
from datetime import datetime
from iointel import Agent, Workflow

# Cargar variables de entorno desde .env
load_dotenv()

# Modelos de datos para las peticiones
class CustomTaskRequest(BaseModel):
    objective: str = Field(..., description="Objetivo o texto base para la tarea", min_length=1, max_length=20000)
    task_name: str = Field(default="custom-task", description="Nombre de la tarea personalizada")
    task_objective: str = Field(..., description="Objetivo específico de la tarea")
    instructions: str = Field(..., description="Instrucciones específicas para el agente")
    agent_name: str = Field(default="Super Agent", description="Nombre del agente")
    agent_instructions: str = Field(default="You are an assistant specialized in neurodivergence, particularly in identifying signs and characteristics of Autism Spectrum Disorder (ASD) and Attention Deficit Hyperactivity Disorder (ADHD). Use psychological principles and evidence-based criteria to provide thoughtful, accurate, and compassionate responses. Always communicate with sensitivity and respect toward neurodiverse individuals.", description="Instrucciones del agente")

class BatchTaskRequest(BaseModel):
    tasks: List[Dict[str, str]] = Field(..., description="Lista de tareas a ejecutar", min_items=1, max_items=20)

class TaskResponse(BaseModel):
    success: bool
    task_id: str
    task_name: str
    objective: str
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    timestamp: str

class BatchTaskResponse(BaseModel):
    success: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    results: List[TaskResponse]
    total_execution_time: Optional[float] = None

# Clase del Super Agente
class IONetSuperAgent:
    def __init__(self, api_key=None):
        # Prioridad: parámetro > .env > variable de entorno del sistema
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Se requiere una API key. Opciones:\n"
                "1. Crear archivo .env con: OPENAI_API_KEY=tu_api_key_aquí\n"
                "2. Configurar variable de entorno: export OPENAI_API_KEY='tu_api_key_aquí'"
            )
        
        self.base_url = os.environ.get("IOINTEL_BASE_URL", "https://api.intelligence.io.solutions/api/v1")
        self.default_model = os.environ.get("IOINTEL_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        
        print(f"✅ API Key configurada: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"🔗 Base URL: {self.base_url}")
        print(f"🤖 Modelo: {self.default_model}")
    
    # You are an assistant specialized in doing anything.
    # You are an assistant specialized in neurodivergence, particularly in identifying signs and characteristics of Autism Spectrum Disorder (ASD) and Attention Deficit Hyperactivity Disorder (ADHD). Use psychological principles and evidence-based criteria to provide thoughtful, accurate, and compassionate responses. Always communicate with sensitivity and respect toward neurodiverse individuals.
    def create_agent(self, name: str = "Super Agent", instructions: str = "You are an assistant specialized in neurodivergence, particularly in identifying signs and characteristics of Autism Spectrum Disorder (ASD) and Attention Deficit Hyperactivity Disorder (ADHD). Use psychological principles and evidence-based criteria to provide thoughtful, accurate, and compassionate responses. Always communicate with sensitivity and respect toward neurodiverse individuals."):
        """Crea un agente personalizado"""
        return Agent(
            name=name,
            instructions=instructions,
            model=self.default_model,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def execute_custom_task(self, 
                                 objective: str, 
                                 task_name: str = "custom-task",
                                 task_objective: str = "Process the given objective",
                                 instructions: str = "Focus on the task requirements",
                                 agent_name: str = "Super Agent",
                                 agent_instructions: str = "You are an assistant specialized in doing anything.") -> Dict[str, Any]:
        """Ejecuta una tarea personalizada"""
        try:
            start_time = datetime.now()
            
            # Crear agente
            agent = self.create_agent(agent_name, agent_instructions)
            
            # Crear workflow
            workflow = Workflow(objective=objective, client_mode=False)
            
            # Ejecutar tarea personalizada
            results = await workflow.custom(
                name=task_name,
                objective=task_objective,
                instructions=instructions,
                agents=[agent]
            ).run_tasks()

            # Mostrar tokens consumidos si están disponibles
            if "usage" in results and "total_tokens" in results["usage"]:
                print(f"🔢 Tokens consumidos: {results['usage']['total_tokens']}")

            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "results": results["results"],
                "execution_time": execution_time,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Error al ejecutar tarea personalizada: {str(e)}")
    
    async def execute_batch_tasks(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Ejecuta múltiples tareas personalizadas"""
        results = []
        
        for i, task in enumerate(tasks, 1):
            try:
                result = await self.execute_custom_task(
                    objective=task.get("objective", ""),
                    task_name=task.get("task_name", f"batch-task-{i}"),
                    task_objective=task.get("task_objective", "Process the given objective"),
                    instructions=task.get("instructions", "Focus on the task requirements"),
                    agent_name=task.get("agent_name", "Super Agent"),
                    agent_instructions=task.get("agent_instructions", "You are an assistant specialized in doing anything.")
                )
                
                results.append({
                    "task_index": i,
                    "status": "success",
                    "task_name": task.get("task_name", f"batch-task-{i}"),
                    "result": result,
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "task_index": i,
                    "status": "error",
                    "task_name": task.get("task_name", f"batch-task-{i}"),
                    "result": None,
                    "error": str(e)
                })
        
        return results

# Crear la aplicación FastAPI
app = FastAPI(
    title="Super Agent API - io.net",
    description="API para ejecutar tareas personalizadas usando Super Agent de io.net",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del super agente
super_agent = None

@app.on_event("startup")
async def startup_event():
    """Inicializa el super agente al arrancar la aplicación"""
    global super_agent
    try:
        super_agent = IONetSuperAgent()
        print("✅ Super Agent inicializado correctamente")
    except Exception as e:
        print(f"❌ Error al inicializar Super Agent: {e}")
        print("📋 Verifica tu configuración:")
        print("   1. Archivo .env con OPENAI_API_KEY")
        print("   2. O variable de entorno OPENAI_API_KEY")

# Endpoints
@app.get("/")
async def root():
    """Endpoint de información de la API"""
    return {
        "service": "Super Agent API - io.net",
        "status": "active",
        "version": "1.0.0",
        "config": {
            "base_url": os.environ.get("IOINTEL_BASE_URL", "https://api.intelligence.io.solutions/api/v1"),
            "model": os.environ.get("IOINTEL_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "dotenv_loaded": True
        },
        "capabilities": [
            "Tareas personalizadas",
            "Procesamiento de texto",
            "Análisis de contenido",
            "Generación de contenido",
            "Tareas en lote"
        ],
        "endpoints": {
            "execute_task": "/execute-task",
            "batch_tasks": "/batch-tasks",
            "quick_task": "/quick-task",
            "analyze_file": "/analyze-file",
            "health": "/health",
            "examples": "/examples"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    global super_agent
    
    # Verificar configuración
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("IOINTEL_BASE_URL", "https://api.intelligence.io.solutions/api/v1")
    model = os.environ.get("IOINTEL_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    
    return {
        "status": "healthy" if super_agent else "unhealthy",
        "agent_ready": super_agent is not None,
        "configuration": {
            "api_key_configured": bool(api_key),
            "api_key_length": len(api_key) if api_key else 0,
            "base_url": base_url,
            "model": model,
            "dotenv_file_exists": os.path.exists(".env")
        },
        "environment_check": {
            "python_dotenv_loaded": True,
            "current_working_dir": os.getcwd(),
            "env_files_checked": [".env", ".env.local", ".env.production"]
        }
    }

@app.post("/execute-task", response_model=TaskResponse)
async def execute_custom_task(request: CustomTaskRequest):
    """
    Ejecuta una tarea personalizada con el Super Agent
    
    - **objective**: Texto base o contexto para la tarea
    - **task_name**: Nombre identificador de la tarea
    - **task_objective**: Objetivo específico de la tarea
    - **instructions**: Instrucciones detalladas para el agente
    - **agent_name**: Nombre del agente (opcional)
    - **agent_instructions**: Instrucciones del agente (opcional)
    """
    global super_agent
    
    if not super_agent:
        raise HTTPException(
            status_code=500,
            detail="Super Agent no está inicializado. Verifica la configuración de la API key en tu archivo .env"
        )
    
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        result = await super_agent.execute_custom_task(
            objective=request.objective,
            task_name=request.task_name,
            task_objective=request.task_objective,
            instructions=request.instructions,
            agent_name=request.agent_name,
            agent_instructions=request.agent_instructions
        )
        
        return TaskResponse(
            success=True,
            task_id=task_id,
            task_name=request.task_name,
            objective=request.objective,
            results=result["results"],
            execution_time=result["execution_time"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        return TaskResponse(
            success=False,
            task_id=task_id,
            task_name=request.task_name,
            objective=request.objective,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.post("/batch-tasks", response_model=BatchTaskResponse)
async def execute_batch_tasks(request: BatchTaskRequest):
    """
    Ejecuta múltiples tareas personalizadas en lote
    
    - **tasks**: Lista de tareas con sus respectivos parámetros
    """
    global super_agent
    
    if not super_agent:
        raise HTTPException(
            status_code=500,
            detail="Super Agent no está inicializado. Verifica la configuración de la API key en tu archivo .env"
        )
    
    start_time = datetime.now()
    
    try:
        batch_results = await super_agent.execute_batch_tasks(request.tasks)
        
        end_time = datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()
        
        # Procesar resultados
        task_responses = []
        completed_tasks = 0
        failed_tasks = 0
        
        for batch_result in batch_results:
            if batch_result["status"] == "success":
                completed_tasks += 1
                task_response = TaskResponse(
                    success=True,
                    task_id=f"batch_{batch_result['task_index']}",
                    task_name=batch_result["task_name"],
                    objective=request.tasks[batch_result["task_index"]-1].get("objective", ""),
                    results=batch_result["result"]["results"],
                    execution_time=batch_result["result"]["execution_time"],
                    timestamp=batch_result["result"]["timestamp"]
                )
            else:
                failed_tasks += 1
                task_response = TaskResponse(
                    success=False,
                    task_id=f"batch_{batch_result['task_index']}",
                    task_name=batch_result["task_name"],
                    objective=request.tasks[batch_result["task_index"]-1].get("objective", ""),
                    error=batch_result["error"],
                    timestamp=datetime.now().isoformat()
                )
            
            task_responses.append(task_response)
        
        return BatchTaskResponse(
            success=True,
            total_tasks=len(request.tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            results=task_responses,
            total_execution_time=total_execution_time
        )
        
    except Exception as e:
        return BatchTaskResponse(
            success=False,
            total_tasks=len(request.tasks),
            completed_tasks=0,
            failed_tasks=len(request.tasks),
            results=[],
            total_execution_time=0
        )

@app.post("/quick-task")
async def quick_task(
    objective: str = Form(..., description="Objetivo o texto base"),
    task: str = Form(..., description="Tarea específica a realizar"),
    instructions: str = Form(default="Process the given objective", description="Instrucciones adicionales")
):
    """
    Ejecuta una tarea rápida usando formulario
    
    - **objective**: Texto base o contexto
    - **task**: Tarea específica a realizar
    - **instructions**: Instrucciones adicionales (opcional)
    """
    global super_agent
    
    if not super_agent:
        raise HTTPException(
            status_code=500,
            detail="Super Agent no está inicializado. Verifica la configuración de la API key en tu archivo .env"
        )
    
    try:
        result = await super_agent.execute_custom_task(
            objective=objective,
            task_name="quick-task",
            task_objective=task,
            instructions=instructions
        )
        
        return {
            "success": True,
            "task": task,
            "objective": objective[:200] + "..." if len(objective) > 200 else objective,
            "results": result["results"],
            "execution_time": result["execution_time"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "task": task,
            "error": str(e)
        }

@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    task: str = Form(..., description="Qué hacer con el archivo"),
    instructions: str = Form(default="Analyze the file content", description="Instrucciones específicas")
):
    """
    Analiza un archivo de texto con una tarea específica
    
    - **file**: Archivo de texto (.txt)
    - **task**: Tarea a realizar con el contenido del archivo
    - **instructions**: Instrucciones específicas
    """
    global super_agent
    
    if not super_agent:
        raise HTTPException(
            status_code=500,
            detail="Super Agent no está inicializado. Verifica la configuración de la API key en tu archivo .env"
        )
    
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten archivos .txt"
        )
    
    try:
        # Leer archivo
        content = await file.read()
        text = content.decode('utf-8')
        
        # Límite configurable desde .env
        max_file_size = int(os.environ.get("MAX_FILE_SIZE", "50000"))
        if len(text) > max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo es demasiado grande (máximo {max_file_size} caracteres)"
            )
        
        # Ejecutar tarea
        result = await super_agent.execute_custom_task(
            objective=text,
            task_name="file-analysis",
            task_objective=task,
            instructions=instructions
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size": len(text),
            "task": task,
            "results": result["results"],
            "execution_time": result["execution_time"]
        }
        
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Error al leer el archivo. Asegúrate de que esté codificado en UTF-8"
        )
    except Exception as e:
        return {
            "success": False,
            "filename": file.filename,
            "task": task,
            "error": str(e)
        }

@app.get("/examples")
async def get_examples():
    """Devuelve ejemplos de uso de la API"""
    return {
        "examples": {
            "text_summarization": {
                "endpoint": "/execute-task",
                "method": "POST",
                "description": "Resumir un texto largo",
                "body": {
                    "objective": "El mercado global de vehículos eléctricos está experimentando un crecimiento exponencial...",
                    "task_name": "summarization",
                    "task_objective": "Resumir el contenido",
                    "instructions": "Crear un resumen conciso de máximo 100 palabras, enfocándose en los puntos clave"
                }
            },
            "content_analysis": {
                "endpoint": "/execute-task",
                "method": "POST",
                "description": "Analizar contenido específico",
                "body": {
                    "objective": "Texto a analizar...",
                    "task_name": "content-analysis",
                    "task_objective": "Analizar el tono y mensaje principal",
                    "instructions": "Identificar el tono, mensaje principal y audiencia objetivo"
                }
            },
            "creative_writing": {
                "endpoint": "/execute-task",
                "method": "POST",
                "description": "Generar contenido creativo",
                "body": {
                    "objective": "Tema: Inteligencia Artificial en la educación",
                    "task_name": "creative-writing",
                    "task_objective": "Escribir un artículo de blog",
                    "instructions": "Crear un artículo de 500 palabras sobre IA en educación, con tono profesional pero accesible"
                }
            },
            "quick_task": {
                "endpoint": "/quick-task",
                "method": "POST",
                "content_type": "application/x-www-form-urlencoded",
                "description": "Tarea rápida usando formulario",
                "body": {
                    "objective": "Datos de ventas del último trimestre...",
                    "task": "Crear un resumen ejecutivo",
                    "instructions": "Enfocarse en tendencias y recomendaciones"
                }
            },
            "batch_processing": {
                "endpoint": "/batch-tasks",
                "method": "POST",
                "description": "Procesar múltiples tareas",
                "body": {
                    "tasks": [
                        {
                            "objective": "Texto 1...",
                            "task_name": "summary-1",
                            "task_objective": "Resumir",
                            "instructions": "Resumen de 50 palabras"
                        },
                        {
                            "objective": "Texto 2...",
                            "task_name": "analysis-1",
                            "task_objective": "Analizar sentimiento",
                            "instructions": "Identificar sentimiento y emociones"
                        }
                    ]
                }
            }
        }
    }

# Manejo de errores globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {
        "success": False,
        "error": f"Error interno del servidor: {str(exc)}",
        "detail": "Verifica la configuración de la API key en tu archivo .env y que el servicio io.net esté disponible"
    }

def check_environment():
    """Verifica la configuración del entorno"""
    print("🔍 Verificando configuración del entorno...")
    
    # Verificar archivo .env
    if os.path.exists(".env"):
        print("✅ Archivo .env encontrado")
    else:
        print("⚠️  Archivo .env no encontrado")
        print("   Crea un archivo .env con:")
        print("   OPENAI_API_KEY=tu_api_key_aquí")
    
    # Verificar API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API Key configurada: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY no configurada")
    
    # Verificar configuración opcional
    base_url = os.environ.get("IOINTEL_BASE_URL")
    if base_url:
        print(f"🔗 Base URL personalizada: {base_url}")
    
    model = os.environ.get("IOINTEL_MODEL")
    if model:
        print(f"🤖 Modelo personalizado: {model}")
    
    max_file_size = os.environ.get("MAX_FILE_SIZE")
    if max_file_size:
        print(f"📁 Tamaño máximo de archivo: {max_file_size} caracteres")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando Super Agent API...")
    check_environment()
    
    print("\n📖 Documentación disponible en: http://localhost:8000/docs")
    print("🔍 Ejemplos de uso en: http://localhost:8000/examples")
    print("❤️  Estado de salud en: http://localhost:8000/health")
    print("💡 Capacidades: Tareas personalizadas, análisis, generación de contenido")
    
    uvicorn.run(
        "super_agent_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
        log_level="info"
    )