import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from datetime import datetime
import faiss
import pickle
import os
import hashlib
from pinecone import Pinecone, ServerlessSpec
import time
import re
import base64
import io
import threading
import tempfile
import sqlite3
from sqlalchemy import create_engine, text, MetaData, inspect
import plotly.express as px
import plotly.graph_objects as go

# Configuration
from config import PINECONE_API_KEY, INDEX_NAME, DIMENSION, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# ================================
# DEPENDENCY CHECKING AND IMPORTS
# ================================

# Check for speech recognition dependencies
try:
    import speech_recognition as sr
    import pyaudio  # Explicitly import to check availability
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ Speech recognition dependencies loaded successfully")
except ImportError as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    print(f"❌ Speech recognition not available: {e}")

# Check for text-to-speech dependencies
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    print("✅ Text-to-speech (gTTS) loaded successfully")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"❌ Text-to-speech not available: {e}")

# Check for pygame (audio playback)
try:
    import pygame
    PYGAME_AVAILABLE = True
    print("✅ Pygame loaded successfully")
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
except ImportError as e:
    PYGAME_AVAILABLE = False
    print(f"❌ Pygame not available: {e}")

# Check for pydub (alternative audio processing)
try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
    print("✅ Pydub loaded successfully")
except ImportError as e:
    PYDUB_AVAILABLE = False
    print(f"❌ Pydub not available: {e}")

# LangChain imports
try:
    from langchain.schema import Document 
    from langchain.vectorstores.base import VectorStore 
    from langchain.embeddings.base import Embeddings
    from langchain.llms.base import LLM
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import Runnable
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.memory import ConversationBufferMemory
    from langchain.chains.conversation.base import ConversationChain
    from pydantic import Field
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain loaded successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"❌ LangChain not available: {e}")

# ================================
# DATABASE SCHEMA ANALYZER
# ================================

class DatabaseSchemaAnalyzer:
    """Analiza y extrae el esquema de la base de datos para generar embeddings"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.engine = None
        self.metadata = None
        self.schema_embeddings = {}
        self.table_descriptions = {}
        
    def connect(self) -> bool:
        """Conecta a la base de datos"""
        try:
            # Crear cadena de conexión
            conn_string = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            self.engine = create_engine(conn_string)
            
            # Probar conexión
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            return True
            
        except Exception as e:
            st.error(f"Error conectando a la base de datos: {e}")
            return False
    
    def extract_schema_info(self) -> Dict[str, Any]:
        """Extrae información detallada del esquema"""
        if not self.engine:
            return {}
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "indexes": {},
            "views": {}
        }
        
        try:
            inspector = inspect(self.engine)
            
            # Obtener tablas
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)
                pk_constraint = inspector.get_pk_constraint(table_name)
                
                schema_info["tables"][table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                    "indexes": indexes,
                    "primary_key": pk_constraint
                }
                
                # Agregar relaciones
                for fk in foreign_keys:
                    schema_info["relationships"].append({
                        "from_table": table_name,
                        "from_column": fk["constrained_columns"],
                        "to_table": fk["referred_table"],
                        "to_column": fk["referred_columns"]
                    })
            
            # Obtener vistas
            for view_name in inspector.get_view_names():
                try:
                    view_definition = inspector.get_view_definition(view_name)
                    schema_info["views"][view_name] = {
                        "definition": view_definition,
                        "columns": inspector.get_columns(view_name)
                    }
                except:
                    pass  # Algunas vistas pueden no ser accesibles
                    
        except Exception as e:
            st.error(f"Error extrayendo esquema: {e}")
        
        return schema_info
    
    def generate_table_descriptions(self, schema_info: Dict) -> Dict[str, str]:
        """Genera descripciones textuales de las tablas para embeddings"""
        descriptions = {}
        
        for table_name, table_info in schema_info["tables"].items():
            description_parts = [f"Tabla: {table_name}"]
            
            # Agregar columnas
            columns_desc = []
            for col in table_info["columns"]:
                col_desc = f"{col['name']} ({col['type']}"
                if col.get('nullable', True) == False:
                    col_desc += ", NOT NULL"
                if col.get('default'):
                    col_desc += f", DEFAULT {col['default']}"
                col_desc += ")"
                columns_desc.append(col_desc)
            
            description_parts.append(f"Columnas: {', '.join(columns_desc)}")
            
            # Agregar clave primaria
            if table_info["primary_key"]["constrained_columns"]:
                pk_cols = ", ".join(table_info["primary_key"]["constrained_columns"])
                description_parts.append(f"Clave primaria: {pk_cols}")
            
            # Agregar claves foráneas
            if table_info["foreign_keys"]:
                fk_descriptions = []
                for fk in table_info["foreign_keys"]:
                    fk_desc = f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}"
                    fk_descriptions.append(fk_desc)
                description_parts.append(f"Claves foráneas: {'; '.join(fk_descriptions)}")
            
            descriptions[table_name] = ". ".join(description_parts)
        
        return descriptions

# ================================
# SQL QUERY GENERATOR
# ================================

class NaturalLanguageSQLGenerator:
    """Genera consultas SQL a partir de lenguaje natural usando embeddings"""
    
    def __init__(self, llm, embeddings_model, schema_analyzer: DatabaseSchemaAnalyzer):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.schema_analyzer = schema_analyzer
        self.schema_embeddings = {}
        self.faiss_index = None
        self.table_names = []
        
    def build_schema_embeddings(self) -> bool:
        """Construye embeddings del esquema de la base de datos"""
        try:
            if not self.schema_analyzer.connect():
                return False
            
            schema_info = self.schema_analyzer.extract_schema_info()
            table_descriptions = self.schema_analyzer.generate_table_descriptions(schema_info)
            
            # Crear embeddings para cada descripción de tabla
            descriptions = []
            table_names = []
            
            for table_name, description in table_descriptions.items():
                descriptions.append(description)
                table_names.append(table_name)
            
            if descriptions:
                embeddings = self.embeddings_model.embed_documents(descriptions)
                
                # Crear índice FAISS
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_array = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                self.faiss_index.add(embeddings_array)
                
                self.table_names = table_names
                self.schema_embeddings = dict(zip(table_names, embeddings))
                
                st.success(f"✅ Embeddings creados para {len(table_names)} tablas")
                return True
            
        except Exception as e:
            st.error(f"Error construyendo embeddings del esquema: {e}")
            return False
    
    def find_relevant_tables(self, query: str, top_k: int = 5, score_threshold: float = 0.4) -> List[str]:
        """Encuentra las tablas más relevantes para una consulta, priorizando scores altos pero asegurando siempre una respuesta"""
        
        if not self.faiss_index:
            st.info("[DEBUG] No se encontró el índice FAISS")
            return []

        try:
            # Generar embedding de la consulta
            st.info("[DEBUG] Generando embedding para la consulta...")
            query_embedding = self.embeddings_model.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)

            # Buscar tablas similares
            st.info("[DEBUG] Buscando tablas similares...")
            top_k = min(top_k, len(self.table_names))
            scores, indices = self.faiss_index.search(query_vector, top_k)

            relevant_tables = []

            for i, idx in enumerate(indices[0]):
                if idx < len(self.table_names):
                    table_name = self.table_names[idx]
                    score = scores[0][i]
                    st.info(f"[DEBUG] Tabla: {table_name}, Score: {score}")
                    relevant_tables.append((table_name, score))

            # Filtrar por score_threshold
            filtered = [table for table in relevant_tables if table[1] >= score_threshold]

            if filtered:
                st.info(f"[DEBUG] Tablas relevantes con score > {score_threshold}: {[t[0] for t in filtered]}")
                return [t[0] for t in sorted(filtered, key=lambda x: x[1], reverse=True)]
            else:
                # Si ninguna tabla pasa el umbral, igual devolver las mejores (ordenadas)
                st.info("[DEBUG] No se encontraron tablas con score alto. Devolviendo las más cercanas disponibles.")
                sorted_tables = sorted(relevant_tables, key=lambda x: x[1], reverse=True)
                return [t[0] for t in sorted_tables]

        except Exception as e:
            st.error(f"[ERROR] Error al procesar la consulta: {str(e)}")
            return []


    def generate_sql_query(self, natural_query: str) -> Tuple[str, List[str], str]:
        """Genera consulta SQL a partir de lenguaje natural"""
        try:
            # Encontrar tablas relevantes
            relevant_tables = self.find_relevant_tables(natural_query, top_k=3)
            
            if not relevant_tables:
                return "", [], "No se encontraron tablas relevantes para la consulta"
            
            # Obtener esquemas detallados de las tablas relevantes
            schema_info = self.schema_analyzer.extract_schema_info()
            detailed_schema = self._build_detailed_schema_context(relevant_tables, schema_info)
            
            # Crear prompt para generar SQL
            sql_prompt = self._create_sql_generation_prompt(natural_query, detailed_schema, relevant_tables)
            
            # Generar SQL usando el LLM
            sql_response = self.llm._call(sql_prompt)
            
            # Extraer y limpiar la consulta SQL
            sql_query = self._extract_sql_from_response(sql_response)
            
            return sql_query, relevant_tables, sql_response
            
        except Exception as e:
            error_msg = f"Error generando consulta SQL: {e}"
            st.error(error_msg)
            return "", [], error_msg
    
    def _build_detailed_schema_context(self, table_names: List[str], schema_info: Dict) -> str:
        """Construye contexto detallado del esquema para las tablas relevantes"""
        schema_context = "ESQUEMA DE BASE DE DATOS:\n\n"
        
        for table_name in table_names:
            if table_name in schema_info["tables"]:
                table_info = schema_info["tables"][table_name]
                schema_context += f"TABLA: {table_name}\n"
                
                # Columnas
                schema_context += "COLUMNAS:\n"
                for col in table_info["columns"]:
                    col_info = f"  - {col['name']}: {col['type']}"
                    if not col.get('nullable', True):
                        col_info += " (NOT NULL)"
                    if col.get('default'):
                        col_info += f" DEFAULT {col['default']}"
                    schema_context += col_info + "\n"
                
                # Clave primaria
                if table_info["primary_key"]["constrained_columns"]:
                    pk_cols = ", ".join(table_info["primary_key"]["constrained_columns"])
                    schema_context += f"CLAVE PRIMARIA: {pk_cols}\n"
                
                # Claves foráneas
                if table_info["foreign_keys"]:
                    schema_context += "CLAVES FORÁNEAS:\n"
                    for fk in table_info["foreign_keys"]:
                        schema_context += f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"
                
                schema_context += "\n"
        
        # Agregar relaciones entre tablas
        relationships = [rel for rel in schema_info["relationships"] 
                        if rel["from_table"] in table_names or rel["to_table"] in table_names]
        
        if relationships:
            schema_context += "RELACIONES:\n"
            for rel in relationships:
                schema_context += f"  - {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
        
        return schema_context
    
    def _create_sql_generation_prompt(self, natural_query: str, schema_context: str, relevant_tables: List[str]) -> str:
        """Crea el prompt para generar SQL"""
        return f"""
Eres un experto en SQL y análisis de bases de datos. Tu tarea es convertir consultas en lenguaje natural a consultas SQL precisas y eficientes.

{schema_context}

CONSULTA EN LENGUAJE NATURAL:
{natural_query}

TABLAS RELEVANTES IDENTIFICADAS:
{', '.join(relevant_tables)}

INSTRUCCIONES:
1. Genera una consulta SQL válida y eficiente para PostgreSQL
2. Usa SOLO las tablas y columnas que aparecen en el esquema proporcionado
3. Incluye JOINs apropiados si es necesario
4. Usa agregaciones cuando sea relevante (COUNT, SUM, AVG, etc.)
5. Incluye filtros WHERE apropiados
6. Ordena los resultados cuando sea lógico
7. Limita los resultados si es apropiado (LIMIT)

FORMATO DE RESPUESTA:
Proporciona SOLO la consulta SQL sin explicaciones adicionales. La consulta debe estar lista para ejecutar.

CONSULTA SQL:
"""

    def _extract_sql_from_response(self, response: str) -> str:
        """Extrae y limpia la consulta SQL de la respuesta del LLM"""
        # Buscar bloques de código SQL
        sql_patterns = [
            r"```sql\s*(.*?)\s*```",
            r"```\s*(SELECT.*?);?\s*```",
            r"(SELECT.*?);?$"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                # Limpiar la consulta
                sql = re.sub(r'\s+', ' ', sql)  # Normalizar espacios
                sql = sql.rstrip(';')  # Remover punto y coma final si existe
                return sql
        
        # Si no se encuentra un patrón, devolver la respuesta limpia
        cleaned_response = response.strip()
        if cleaned_response.upper().startswith('SELECT'):
            return cleaned_response
        
        return ""

# ================================
# DATABASE QUERY EXECUTOR
# ================================

class DatabaseQueryExecutor:
    """Ejecuta consultas SQL y formatea resultados"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        
    def execute_query(self, sql_query: str, limit: int = 100) -> Tuple[pd.DataFrame, str, bool]:
        """Ejecuta una consulta SQL y retorna los resultados"""
        try:
            # Agregar LIMIT si no existe y es una consulta SELECT
            if sql_query.upper().strip().startswith('SELECT') and 'LIMIT' not in sql_query.upper():
                sql_query = f"{sql_query} LIMIT {limit}"
            
            # Crear conexión
            conn = psycopg2.connect(**self.connection_params)
            
            # Ejecutar consulta
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            success_msg = f"✅ Consulta ejecutada exitosamente. {len(df)} filas obtenidas."
            return df, success_msg, True
            
        except Exception as e:
            error_msg = f"❌ Error ejecutando consulta: {str(e)}"
            return pd.DataFrame(), error_msg, False
    
    def format_results_for_voice(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Formatea los resultados para síntesis de voz"""
        if df.empty:
            return "No se encontraron resultados para tu consulta."
        
        total_rows = len(df)
        rows_to_describe = min(max_rows, total_rows)
        
        # Introducción
        result_text = f"Encontré {total_rows} resultado{'s' if total_rows != 1 else ''}. "
        
        if total_rows > max_rows:
            result_text += f"Te voy a mostrar los primeros {rows_to_describe}. "
        
        # Describir columnas
        columns = list(df.columns)
        if len(columns) <= 3:
            result_text += f"Las columnas son: {', '.join(columns)}. "
        else:
            result_text += f"Hay {len(columns)} columnas principales. "
        
        # Describir filas
        for i, row in df.head(rows_to_describe).iterrows():
            row_text = f"Registro {i+1}: "
            row_parts = []
            
            for col in columns[:4]:  # Máximo 4 columnas por fila para voz
                value = row[col]
                if pd.isna(value):
                    continue
                    
                # Formatear valor según tipo
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    row_parts.append(f"{col}: {value}")
                else:
                    # Truncar strings muy largos
                    str_value = str(value)
                    if len(str_value) > 30:
                        str_value = str_value[:30] + "..."
                    row_parts.append(f"{col}: {str_value}")
            
            if row_parts:
                result_text += row_text + ", ".join(row_parts) + ". "
        
        # Resumen estadístico si es apropiado
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0 and len(df) > 1:
            summary_parts = []
            for col in numeric_columns[:2]:  # Máximo 2 columnas numéricas
                mean_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                
                if isinstance(mean_val, float) and mean_val.is_integer():
                    mean_val = int(mean_val)
                if isinstance(max_val, float) and max_val.is_integer():
                    max_val = int(max_val)
                if isinstance(min_val, float) and min_val.is_integer():
                    min_val = int(min_val)
                
                summary_parts.append(f"{col}: promedio {mean_val}, máximo {max_val}, mínimo {min_val}")
            
            if summary_parts:
                result_text += f"Resumen: {'. '.join(summary_parts)}. "
        
        return result_text
    
    def create_visualization(self, df: pd.DataFrame, query_type: str = "auto") -> Optional[go.Figure]:
        """Crea visualización automática basada en los datos"""
        if df.empty or len(df.columns) < 2:
            return None
        
        try:
            # Detectar tipos de columnas
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Lógica de visualización automática
            if len(numeric_columns) >= 2:
                # Scatter plot o correlation heatmap
                if len(df) <= 1000:  # Scatter para datasets pequeños
                    fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], 
                                   title=f"Relación entre {numeric_columns[0]} y {numeric_columns[1]}")
                    return fig
            
            if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
                # Bar chart
                if len(df[categorical_columns[0]].unique()) <= 20:  # No más de 20 categorías
                    fig = px.bar(df, x=categorical_columns[0], y=numeric_columns[0],
                               title=f"{numeric_columns[0]} por {categorical_columns[0]}")
                    return fig
            
            if len(datetime_columns) >= 1 and len(numeric_columns) >= 1:
                # Time series
                fig = px.line(df, x=datetime_columns[0], y=numeric_columns[0],
                            title=f"Evolución de {numeric_columns[0]} en el tiempo")
                return fig
            
            if len(categorical_columns) >= 1:
                # Pie chart para una columna categórica
                value_counts = df[categorical_columns[0]].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribución de {categorical_columns[0]}")
                return fig
            
            if len(numeric_columns) >= 1:
                # Histogram para columna numérica
                fig = px.histogram(df, x=numeric_columns[0], 
                                 title=f"Distribución de {numeric_columns[0]}")
                return fig
                
        except Exception as e:
            st.warning(f"No se pudo crear visualización: {e}")
            return None

# ================================
# ENHANCED VOICE PROCESSOR (from original)
# ================================

class VoiceProcessor:
    """Handles speech-to-text and text-to-speech operations with better error handling"""
    
    def __init__(self):
        self.speech_available = SPEECH_RECOGNITION_AVAILABLE
        self.tts_available = TTS_AVAILABLE
        self.audio_available = PYGAME_AVAILABLE or PYDUB_AVAILABLE
        self.is_listening = False
        
        if self.speech_available:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("✅ Speech recognition initialized successfully")
            except Exception as e:
                print(f"⚠️ Speech recognition initialization failed: {e}")
                self.speech_available = False
        
        if PYGAME_AVAILABLE:
            try:
                self.audio_backend = 'pygame'
                print("✅ Audio playback (pygame) initialized successfully")
            except Exception as e:
                print(f"⚠️ Pygame audio initialization failed: {e}")
                if PYDUB_AVAILABLE:
                    self.audio_backend = 'pydub'
                else:
                    self.audio_available = False
        elif PYDUB_AVAILABLE:
            self.audio_backend = 'pydub'
        else:
            self.audio_available = False
    
    def get_status(self) -> Dict[str, bool]:
        return {
            'speech_recognition': self.speech_available,
            'text_to_speech': self.tts_available,
            'audio_playback': self.audio_available,
            'fully_functional': all([self.speech_available, self.tts_available, self.audio_available])
        }
    
    def listen_to_speech(self, timeout: int = 10, phrase_timeout: int = 5) -> Optional[str]:
        if not self.speech_available:
            return "Error: Reconocimiento de voz no disponible."
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            with self.microphone as source:
                st.info("🎤 Escuchando... Habla ahora")
                self.is_listening = True
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
                self.is_listening = False
                st.info("🔄 Procesando audio...")
                
                try:
                    text = self.recognizer.recognize_google(audio, language='es-ES')
                    return text
                except sr.UnknownValueError:
                    try:
                        text = self.recognizer.recognize_google(audio, language='es-MX')
                        return text
                    except sr.UnknownValueError:
                        return None
                
        except sr.WaitTimeoutError:
            self.is_listening = False
            st.warning("⏰ Tiempo de espera agotado. No se detectó voz.")
            return None
        except Exception as e:
            self.is_listening = False
            st.error(f"❌ Error en reconocimiento de voz: {e}")
            return None
    
    def text_to_speech(self, text: str, lang: str = 'es') -> bool:
        if not self.tts_available or not self.audio_available:
            return False
        
        try:
            clean_text = self._clean_text_for_speech(text)
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                tts.save(temp_file.name)
                temp_filename = temp_file.name
            
            success = self._play_audio_file(temp_filename)
            
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            return success
            
        except Exception as e:
            st.error(f"❌ Error en text-to-speech: {e}")
            return False
    
    def _play_audio_file(self, filename: str) -> bool:
        try:
            if self.audio_backend == 'pygame' and PYGAME_AVAILABLE:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            elif self.audio_backend == 'pydub' and PYDUB_AVAILABLE:
                audio = AudioSegment.from_mp3(filename)
                play(audio)
                return True
            return False
        except Exception as e:
            st.error(f"❌ Error reproduciendo audio: {e}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        # Clean text for better speech synthesis
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        
        replacements = {
            '€': 'euros', '$': 'dólares', '%': 'por ciento', '&': 'y',
            '@': 'arroba', '#': 'numeral', '+': 'más', '=': 'igual',
            '<': 'menor que', '>': 'mayor que', '\n': '. ', '\t': ' '
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ================================
# CUSTOM LANGCHAIN LLM WRAPPER
# ================================

class DeepSeekLLM(LLM):
    """Custom LangChain LLM wrapper for DeepSeek API"""
    
    api_key: str = Field(...)
    base_url: str = Field(default="https://api.deepseek.com/v1")
    model_name: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1000)
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the DeepSeek API"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"

# ================================
# CUSTOM LANGCHAIN EMBEDDINGS
# ================================

class SentenceTransformerEmbeddings(Embeddings):
    """Custom LangChain embeddings wrapper for SentenceTransformers"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.model.encode([text])
        return embedding[0].tolist()

# ================================
# INTELLIGENT DATABASE ASSISTANT
# ================================

class IntelligentDatabaseAssistant:
    """Asistente inteligente que combina SQL generation con respuestas conversacionales"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        
        # Inicializar componentes
        self.embeddings_model = SentenceTransformerEmbeddings()
        self.llm = DeepSeekLLM(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        # Inicializar análisis de esquema y generación SQL
        self.schema_analyzer = DatabaseSchemaAnalyzer(connection_params)
        self.sql_generator = NaturalLanguageSQLGenerator(
            self.llm, self.embeddings_model, self.schema_analyzer
        )
        self.query_executor = DatabaseQueryExecutor(connection_params)
        
        # Vector store para contexto de conversación
        self.conversation_memory = []
        self.schema_ready = False
        
        # Memoria de conversación con LangChain
        if LANGCHAIN_AVAILABLE:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
    def initialize_schema(self) -> bool:
        """Inicializa el análisis del esquema y embeddings"""
        try:
            st.info("🔄 Analizando esquema de base de datos...")
            success = self.sql_generator.build_schema_embeddings()
            if success:
                self.schema_ready = True
                st.success("✅ Esquema de base de datos analizado correctamente")
                return True
            else:
                st.error("❌ Error inicializando esquema de base de datos")
                return False
        except Exception as e:
            st.error(f"❌ Error en inicialización: {e}")
            return False
    
    def process_query(self, user_query: str) -> Tuple[str, pd.DataFrame, Optional[go.Figure], str]:
        """Procesa una consulta del usuario y retorna respuesta completa"""
        
        if not self.schema_ready:
            error_msg = "⚠️ El esquema de la base de datos no está inicializado. Por favor, configura la conexión primero."
            st.error(error_msg)  # Mostrar error en el frontend
            return error_msg, pd.DataFrame(), None, ""
        
        try:
            # Mostrar información sobre el proceso
            st.info(f"[DEBUG] Procesando consulta: {user_query}")
            
            # Clasificar tipo de consulta
            # query_type = self._classify_query(user_query)
            # st.info(f"[DEBUG] Tipo de consulta identificado: {query_type}")
            
            # if query_type == "data_query":
            #     st.info("[DEBUG] Procesando consulta de datos...")
            #     return self._process_data_query(user_query)
            # elif query_type == "schema_question":
            #     st.info("[DEBUG] Procesando pregunta sobre el esquema...")
            #     return self._process_schema_question(user_query)
            # elif query_type == "general_conversation":
            #     st.info("[DEBUG] Procesando conversación general...")
            #     return self._process_general_conversation(user_query)
            # else:
            #     st.info("[DEBUG] Procesando consulta de datos por defecto...")
            return self._process_data_query(user_query)
                    
        except Exception as e:
            error_msg = f"❌ Error procesando consulta: {e}"
            st.error(error_msg)  # Mostrar el error en el frontend
            return error_msg, pd.DataFrame(), None, ""
    
    def _classify_query(self, query: str) -> str:
        """Clasifica el tipo de consulta del usuario"""
        query_lower = query.lower()
        
        # Palabras clave para consultas de datos
        data_keywords = [
            'mostrar', 'buscar', 'encontrar', 'listar', 'cuántos', 'cuántas',
            'total', 'suma', 'promedio', 'máximo', 'mínimo', 'contar',
            'filtrar', 'donde', 'que tengan', 'con', 'sin', 'entre',
            'mayor', 'menor', 'igual', 'diferente', 'últimos', 'primeros'
        ]
        
        # Palabras clave para preguntas sobre esquema
        schema_keywords = [
            'qué tablas', 'qué columnas', 'estructura', 'esquema',
            'qué campos', 'qué información', 'qué datos tienes',
            'describe', 'explica la tabla', 'relaciones'
        ]
        
        # Contar coincidencias
        data_score = sum(1 for keyword in data_keywords if keyword in query_lower)
        schema_score = sum(1 for keyword in schema_keywords if keyword in query_lower)
        
        if schema_score > data_score:
            return "schema_question"
        elif data_score > 0:
            return "data_query"
        else:
            return "general_conversation"
    
    def _is_small_talk(self, user_query: str) -> bool:
        """Usa el LLM para determinar si el mensaje es charla informal (no requiere SQL)"""
        prompt = f"""
        Clasifica el siguiente mensaje como 'charla' o 'consulta SQL'.
        Solo responde una palabra: 'charla' o 'consulta'.
        
        Mensaje: "{user_query}"
        """
        try:
            response = self.llm(prompt).strip().lower()
            return "charla" in response
        except Exception as e:
            # Fallback si hay error, asumimos que no es charla
            return False


    def _process_data_query(self, user_query: str) -> Tuple[str, pd.DataFrame, Optional[go.Figure], str]:
        """Procesa consultas que requieren datos de la base de datos o conversación informal."""

        # Paso 1: Detectar si es solo charla informal
        if self._is_small_talk(user_query):
            respuesta = self.llm(f"Responde de forma amable a este mensaje: {user_query}")
            # Registrar en la memoria conversacional si corresponde
            self._add_to_memory(user_query, respuesta, "SELECT 'Charla informal' AS respuesta;")
            return respuesta, pd.DataFrame(), None, "SELECT 'Charla informal' AS respuesta;"

        # Paso 2: Generar consulta SQL
        sql_query, relevant_tables, sql_response = self.sql_generator.generate_sql_query(user_query)

        if not sql_query:
            error_msg = "❌ No se pudo generar una consulta SQL válida para tu pregunta."
            return error_msg, pd.DataFrame(), None, ""

        # Paso 3: Ejecutar la consulta SQL
        df_result, exec_message, success = self.query_executor.execute_query(sql_query)

        if not success:
            return exec_message, pd.DataFrame(), None, sql_query

        # Paso 4: Generar respuesta conversacional basada en los datos obtenidos
        conversational_response = self._generate_conversational_response(
            user_query, df_result, sql_query, relevant_tables
        )

        # Paso 5: Crear visualización si aplica
        visualization = self.query_executor.create_visualization(df_result)

        # Paso 6: Registrar en memoria
        self._add_to_memory(user_query, conversational_response, sql_query)

        return conversational_response, df_result, visualization, sql_query

    
    def _process_schema_question(self, user_query: str) -> Tuple[str, pd.DataFrame, Optional[go.Figure], str]:
        """Procesa preguntas sobre el esquema de la base de datos"""
        
        schema_info = self.schema_analyzer.extract_schema_info()
        
        # Crear contexto del esquema
        schema_context = self._build_schema_summary(schema_info)
        
        # Generar respuesta usando LLM
        prompt = f"""
Eres un asistente experto en bases de datos. El usuario tiene una pregunta sobre el esquema de la base de datos.

ESQUEMA DE LA BASE DE DATOS:
{schema_context}

PREGUNTA DEL USUARIO:
{user_query}

Responde de manera clara y conversacional, explicando la estructura de la base de datos según la pregunta del usuario.
Si el usuario pregunta sobre tablas específicas, describe sus columnas y relaciones.
Si pregunta sobre qué datos están disponibles, proporciona un resumen útil.
"""
        
        response = self.llm._call(prompt)
        
        return response, pd.DataFrame(), None, ""
    
    def _process_general_conversation(self, user_query: str) -> Tuple[str, pd.DataFrame, Optional[go.Figure], str]:
        """Procesa conversación general relacionada con la base de datos"""
        
        # Obtener contexto de conversaciones anteriores
        conversation_context = self._get_conversation_context()
        
        prompt = f"""
Eres un asistente inteligente especializado en bases de datos. Puedes ayudar a los usuarios a entender y consultar sus datos.

CONTEXTO DE CONVERSACIÓN ANTERIOR:
{conversation_context}

MENSAJE DEL USUARIO:
{user_query}

Responde de manera amigable y útil. Si el usuario necesita información específica de la base de datos, sugiérele cómo puede preguntarlo.
Si es una pregunta general, responde apropiadamente manteniendo el contexto de que eres un asistente de base de datos.
"""
        
        response = self.llm._call(prompt)
        
        return response, pd.DataFrame(), None, ""
    
    def _generate_conversational_response(
        self, 
        user_query: str, 
        df_result: pd.DataFrame, 
        sql_query: str, 
        relevant_tables: List[str]
    ) -> str:
        """Genera una respuesta conversacional basada en los resultados"""
        
        if df_result.empty:
            return "No encontré resultados para tu consulta. Puedes intentar reformular la pregunta o verificar los criterios de búsqueda."
        
        # Preparar resumen de datos
        data_summary = self._create_data_summary(df_result)
        
        # Crear prompt para respuesta conversacional
        prompt = f"""
Eres un asistente de datos amigable y conversacional. El usuario hizo esta pregunta sobre la base de datos:

PREGUNTA ORIGINAL: {user_query}

CONSULTA SQL GENERADA: {sql_query}

TABLAS CONSULTADAS: {', '.join(relevant_tables)}

RESUMEN DE RESULTADOS:
{data_summary}

INSTRUCCIONES:
1. Responde de manera conversacional y amigable
2. Explica los hallazgos principales de manera clara
3. Menciona números específicos y datos relevantes
4. No menciones la consulta SQL técnica a menos que sea relevante
5. Si hay patrones interesantes en los datos, destacalos
6. Mantén un tono profesional pero accesible

Genera una respuesta que explique los resultados de manera comprensible:
"""
        
        response = self.llm._call(prompt)
        
        return response
    
    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Crea un resumen de los datos para el LLM"""
        if df.empty:
            return "No hay datos disponibles."
        
        summary_parts = []
        
        # Información básica
        summary_parts.append(f"Total de registros: {len(df)}")
        summary_parts.append(f"Columnas: {', '.join(df.columns.tolist())}")
        
        # Primeras filas (sample)
        if len(df) > 0:
            sample_size = min(3, len(df))
            summary_parts.append("\nPrimeros registros:")
            for i, row in df.head(sample_size).iterrows():
                row_desc = []
                for col, val in row.items():
                    if pd.notna(val):
                        row_desc.append(f"{col}: {val}")
                summary_parts.append(f"- {', '.join(row_desc[:4])}")  # Max 4 campos por fila
        
        # Estadísticas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nEstadísticas numéricas:")
            for col in numeric_cols[:3]:  # Max 3 columnas numéricas
                stats = df[col].describe()
                summary_parts.append(f"- {col}: promedio {stats['mean']:.2f}, min {stats['min']}, max {stats['max']}")
        
        return "\n".join(summary_parts)
    
    def _build_schema_summary(self, schema_info: Dict) -> str:
        """Construye un resumen del esquema para respuestas conversacionales"""
        summary_parts = []
        
        # Tablas disponibles
        tables = list(schema_info["tables"].keys())
        summary_parts.append(f"TABLAS DISPONIBLES ({len(tables)}): {', '.join(tables)}")
        
        # Descripción de cada tabla
        for table_name, table_info in schema_info["tables"].items():
            columns = [col['name'] for col in table_info["columns"]]
            summary_parts.append(f"\nTABLA {table_name}:")
            summary_parts.append(f"  Columnas ({len(columns)}): {', '.join(columns)}")
            
            # Clave primaria
            if table_info["primary_key"]["constrained_columns"]:
                pk = ', '.join(table_info["primary_key"]["constrained_columns"])
                summary_parts.append(f"  Clave primaria: {pk}")
        
        # Relaciones
        if schema_info["relationships"]:
            summary_parts.append(f"\nRELACIONES ({len(schema_info['relationships'])}):")
            for rel in schema_info["relationships"][:5]:  # Max 5 relaciones
                summary_parts.append(f"  {rel['from_table']} -> {rel['to_table']}")
        
        return "\n".join(summary_parts)
    
    def _add_to_memory(self, query: str, response: str, sql_query: str = ""):
        """Agrega interacción a la memoria de conversación"""
        self.conversation_memory.append({
            "timestamp": datetime.now(),
            "user_query": query,
            "response": response,
            "sql_query": sql_query
        })
        
        # Mantener solo las últimas 10 interacciones
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
    
    def _get_conversation_context(self) -> str:
        """Obtiene contexto de conversaciones anteriores"""
        if not self.conversation_memory:
            return "No hay conversación anterior."
        
        context_parts = []
        for interaction in self.conversation_memory[-3:]:  # Últimas 3 interacciones
            context_parts.append(f"Usuario: {interaction['user_query']}")
            context_parts.append(f"Asistente: {interaction['response'][:200]}...")  # Truncar respuesta larga
        
        return "\n".join(context_parts)

# ================================
# STREAMLIT APPLICATION
# ================================

def main():
    st.set_page_config(
        page_title="Asistente Inteligente de Base de Datos",
        page_icon="🗃️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🗃️ Asistente Inteligente de Base de Datos")
    st.markdown("Consulta tu base de datos usando lenguaje natural y recibe respuestas inteligentes")
    
    # Inicializar estado de sesión
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
        st.session_state.conversation_history = []
        st.session_state.voice_processor = VoiceProcessor()
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Estado del sistema de voz
        voice_status = st.session_state.voice_processor.get_status()
        
        st.subheader("🎤 Estado del Sistema de Voz")
        status_icon = "✅" if voice_status['fully_functional'] else "⚠️"
        st.markdown(f"{status_icon} **Estado General:** {'Funcional' if voice_status['fully_functional'] else 'Parcial'}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"🎤 Reconocimiento: {'✅' if voice_status['speech_recognition'] else '❌'}")
            st.markdown(f"🔊 Audio: {'✅' if voice_status['audio_playback'] else '❌'}")
        with col2:
            st.markdown(f"🗣️ Síntesis: {'✅' if voice_status['text_to_speech'] else '❌'}")
        
        st.divider()
        
        # Configuración de base de datos
        st.subheader("🗄️ Configuración de Base de Datos")
        
        db_host = st.text_input("Host", value="localhost", key="db_host")
        db_port = st.number_input("Puerto", value=5432, key="db_port")
        db_name = st.text_input("Base de Datos", value="", key="db_name")
        db_user = st.text_input("Usuario", value="", key="db_user")
        db_password = st.text_input("Contraseña", type="password", key="db_password")
        
        if st.button("🔌 Conectar y Analizar Esquema"):
            if all([db_host, db_port, db_name, db_user, db_password]):
                connection_params = {
                    'host': db_host,
                    'port': int(db_port),
                    'database': db_name,
                    'user': db_user,
                    'password': db_password
                }
                
                with st.spinner("Conectando y analizando esquema..."):
                    st.session_state.assistant = IntelligentDatabaseAssistant(connection_params)
                    if st.session_state.assistant.initialize_schema():
                        st.success("✅ ¡Conexión exitosa! El asistente está listo.")
                    else:
                        st.error("❌ Error en la conexión o análisis del esquema")
                        st.session_state.assistant = None
            else:
                st.error("❌ Por favor completa todos los campos")
    
    # Área principal
    if st.session_state.assistant is None:
        st.info("👈 Por favor configura la conexión a la base de datos en el panel lateral para comenzar.")
        return
    
    # Interfaz de consulta
    st.subheader("💬 Consulta tu Base de Datos")
    
    # Métodos de entrada
    input_method = st.radio(
        "Método de entrada:",
        ["✏️ Texto", "🎤 Voz"] if voice_status['speech_recognition'] else ["✏️ Texto"],
        horizontal=True
    )
    
    user_query = ""
    
    if input_method == "✏️ Texto":
        user_query = st.text_area(
            "Escribe tu consulta:",
            placeholder="Ejemplo: ¿Cuántos usuarios registrados tengo este mes?",
            height=100
        )
        query_button = st.button("🔍 Consultar", type="primary")
    
    else:  # Voz
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎤 Iniciar Grabación", type="primary"):
                with st.spinner("Escuchando..."):
                    recognized_text = st.session_state.voice_processor.listen_to_speech()
                    if recognized_text:
                        st.session_state.recognized_query = recognized_text
                        st.success(f"✅ Reconocido: {recognized_text}")
                    else:
                        st.error("❌ No se pudo reconocer el audio")
        
        with col2:
            if 'recognized_query' in st.session_state:
                user_query = st.text_area(
                    "Consulta reconocida (editable):",
                    value=st.session_state.recognized_query,
                    height=100
                )
            else:
                st.info("Presiona el botón de grabación para comenzar")
        
        query_button = st.button("🔍 Procesar Consulta", type="primary") if user_query else False
    
    # Procesar consulta
    if query_button and user_query.strip():
        with st.spinner("🤔 Analizando tu consulta..."):
            response, df_result, visualization, sql_query = st.session_state.assistant.process_query(user_query)
            print(f"respuesta: {response}\n")
            print(f"df_result: {df_result}\n")
            print(f"visualization: {visualization}\n")
            print(f"sql_query: {sql_query}\n")
            
            # Agregar a historial
            st.session_state.conversation_history.append({
                "query": user_query,
                "response": response,
                "data": df_result,
                "visualization": visualization,
                "sql": sql_query,
                "timestamp": datetime.now()
            })

        # Mostrar respuesta
        st.subheader("🤖 Respuesta del Asistente")
        st.markdown(response)

        # 🔊 Síntesis de voz automática
        if voice_status['text_to_speech'] and voice_status['audio_playback']:
            try:
                with st.spinner("🎙️ Reproduciendo respuesta..."):
                    st.info(f"🔊 Reproduciendo respuesta por voz... {df_result}")
                    voice_text = (
                        st.session_state.voice_processor.format_results_for_voice(df_result)
                        if not df_result.empty else response
                    )
                    success = st.session_state.voice_processor.text_to_speech(voice_text)
                    if not success:
                        st.error("❌ Error reproduciendo audio (fallo interno del sintetizador)")
            except Exception as e:
                st.error(f"❌ Ocurrió un error durante la síntesis de voz: {str(e)}")
        else:
            st.info("ℹ️ La reproducción por voz está desactivada. Actívala en la configuración para escuchar la respuesta.")


        # Mostrar datos si existen
        if not df_result.empty:
            st.subheader("📊 Datos Obtenidos")
            
            # Pestañas para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["📋 Tabla", "📈 Visualización", "💻 SQL"])
            
            with tab1:
                st.dataframe(df_result, use_container_width=True)
                st.caption(f"Mostrando {len(df_result)} registros")
            
            with tab2:
                if visualization:
                    st.plotly_chart(visualization, use_container_width=True)
                else:
                    st.info("No se pudo generar una visualización automática para estos datos")
            
            with tab3:
                if sql_query:
                    st.code(sql_query, language="sql")
                else:
                    st.info("No se generó consulta SQL para esta respuesta")
    
    # Historial de conversación
    if st.session_state.conversation_history:
        st.subheader("📝 Historial de Conversación")
        
        for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Últimas 5
            with st.expander(f"💬 {item['query'][:60]}... - {item['timestamp'].strftime('%H:%M:%S')}"):
                st.markdown(f"**Pregunta:** {item['query']}")
                st.markdown(f"**Respuesta:** {item['response']}")
                
                if not item['data'].empty:
                    st.markdown(f"**Datos:** {len(item['data'])} registros obtenidos")
                    if st.checkbox(f"Ver datos #{len(st.session_state.conversation_history)-i}", key=f"show_data_{i}"):
                        st.dataframe(item['data'])
                
                if item['sql']:
                    if st.checkbox(f"Ver SQL #{len(st.session_state.conversation_history)-i}", key=f"show_sql_{i}"):
                        st.code(item['sql'], language="sql")
        
        # Botón para limpiar historial
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.conversation_history = []
            st.rerun()

if __name__ == "__main__":
    main()