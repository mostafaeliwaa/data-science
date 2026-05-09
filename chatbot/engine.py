import pandas as pd
import google.generativeai as genai
import re
import io
import numpy as np

class MarketingAnalyst:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        self.model_name = self._get_best_available_model()
        print(f"🚀 AI Model Selected: {self.model_name}")
        self.model = genai.GenerativeModel(self.model_name)
        
        self.system_instruction = """
        Act as a recommender and analyst at marketing . Analyze the pandas Dataframe `df` to answer the user query. give recommendations to the user to get best results.
        
        ### RULES:
        1. **Output:** Assign final answer to variable `result`.
        2. **Format:** `result` MUST be a `pd.DataFrame` (for data) or `str` (for text). NEVER return single numbers.
        3. **Filter:** Always run `df = df[df['campaign_name'].notna()]` first.
        4. **Language:** Reply in the SAME language as the user (Arabic/English).
        
        ### METRICS MAPPING:
        - Spend -> `amount_spent_...`
        - Results -> `results`
        - CPA -> `amount_spent_... / results`
        - ROI -> `((conversion_value - amount_spent) / amount_spent) * 100`
        - CTR -> `ctr_...`
        
        ### CONTEXT:
        - If query implies context (e.g. "what about CPA?"), use previous history.
        - Use `pd` and `np` libraries. No loops.
        """

    def _get_best_available_model(self):
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            for m in available_models:
                if 'flash' in m.lower(): return m
            
            for m in available_models:
                if 'pro' in m.lower() and 'vision' not in m.lower(): return m
            
            return available_models[0] if available_models else 'gemini-pro'
        except:
            return 'gemini-pro'

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(c).strip().lower().replace(" ", "_").replace(r"[().]", "").replace("/", "_per_") for c in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,EGP%]', '', regex=True), errors='ignore')
                except: pass
        for col in df.columns:
            if any(x in col for x in ['date', 'start', 'end']):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        return match.group(1) if match else text.replace("```", "")

    def analyze(self, dataframe, user_query, chat_history=[]):
        try:
            schema = "\n".join([f"- {col} ({dtype})" for col, dtype in zip(dataframe.columns, dataframe.dtypes)])
            sample = dataframe.head(2).to_string()

            history_text = ""
            if chat_history:
                history_text = "### Previous Conversation:\n"
                for msg in chat_history[-6:]:
                    role = "User" if msg['role'] == 'user' else "AI"
                    content = str(msg['content'])
                    
                    if role == "AI" and len(content) > 800:
                        content = content[:800] + "... [Content Truncated]"
                    
                    history_text += f"- {role}: {content}\n"

            full_prompt = f"""
            {self.system_instruction}

            ### DATA SCHEMA:
            {schema}
            
            ### SAMPLE:
            {sample}

            {history_text}

            ### QUERY:
            {user_query}
            
            Code:
            """

            response = self.model.generate_content(full_prompt)
            generated_code = self._extract_code(response.text)

            local_vars = {"df": dataframe, "pd": pd, "np": np}
            exec(generated_code, {}, local_vars)

            if "result" in local_vars:
                result = local_vars["result"]
                if isinstance(result, pd.DataFrame):
                    return result.head(50).to_dict(orient="records")
                return str(result)
            else:
                return "Error: No result returned."

        except Exception as e:
            return f"Analysis Error: {str(e)}"