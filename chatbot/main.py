from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from .engine import MarketingAnalyst 
import pandas as pd
import io
import os
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Marketing Campaign Chatbot")

# قراءة المفتاح
API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================
# 💾 THE MEMORY STORE (الرامات)
# ============================================================
SESSION_STORE = {}

# تهيئة المحرك
analyst = MarketingAnalyst(API_KEY)

@app.get("/")
def home():
    return {"status": "Online", "sessions_active": len(SESSION_STORE)}

# ------------------------------------------------------------
# 1. Endpoint لرفع الملف (بيشتغل مرة واحدة في الأول)
# ------------------------------------------------------------
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key is missing")
        
    try:
        # قراءة وتجهيز الملف
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # تنظيف الداتا فوراً
        df_clean = analyst._preprocess_data(df)
        
        # إنشاء جلسة جديدة
        session_id = str(uuid.uuid4())
        
        # تخزين الداتا + هيستوري فاضي
        SESSION_STORE[session_id] = {
            "df": df_clean,
            "history": []
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "File uploaded and cached. Use session_id to chat.",
            "columns": list(df_clean.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 2. Endpoint للشات (سريع جداً + ذاكرة)
# ------------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(
    query: str = Body(..., embed=True), 
    session_id: str = Body(..., embed=True)
):
    # التحقق من الجلسة
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session ID not found. Please upload file first.")
    
    try:
        # استدعاء البيانات من الرامات
        session_data = SESSION_STORE[session_id]
        df = session_data['df']
        history = session_data['history']
        
        # التحليل (مع تمرير الهيستوري)
        result = analyst.analyze(df, query, chat_history=history)
        
        # تحديث الذاكرة
        result_str = str(result) if not isinstance(result, list) else "Result Table (Dataframe)"
        
        session_data['history'].append({"role": "user", "content": query})
        session_data['history'].append({"role": "assistant", "content": result_str})
        
        # نحافظ على آخر 20 رسالة (Sliding Window)
        if len(session_data['history']) > 20:
             session_data['history'] = session_data['history'][-20:]
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        return {"error": str(e)}

# تنظيف الذاكرة
@app.delete("/clear_session")
async def clear_session(session_id: str):
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return {"status": "success", "message": "Session cleared."}
    return {"status": "error", "message": "Session ID not found."}