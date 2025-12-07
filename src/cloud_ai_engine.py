from PyQt5.QtCore import QThread, pyqtSignal
import io, base64, requests, re, json

# ==========================================================
# WORKER 1: For the initial, detailed analysis
# ==========================================================
class CombinedAIWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, segmented_pil_image):
        super().__init__()
        self.image_data = segmented_pil_image
        self.api_key = "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"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-3-haiku"

    def run(self):
        try:
            buffer = io.BytesIO()
            self.image_data.save(buffer, format="PNG")
            data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

            # --- YOUR IMPROVED PROMPT ---
            prompt = """
            You are an expert radiologist writing a formal medical report for an oncologist based on the provided brain MRI scan.

            Structure your response using Markdown with the following sections:
            1.  **### RADIOLOGY REPORT**
            2.  **#### Key Findings:** (Use a bulleted list to describe the lesion's location, size, morphology, and any mass effect.)
            3.  **#### Differential Diagnosis:** (Provide a prioritized list of possible diagnoses, e.g., Glioblastoma, Metastasis, Abscess.)
            4.  **#### Impression & Recommendation:** (Give a concise final impression and recommend the next clinical steps.)

            In a separate section at the very end, return a raw JSON object for the system with your best estimate for the following parameters. Do not add any text before or after this JSON block.

            ```json
            {
              "classification": "String (Your top diagnosis, e.g., 'Glioma', 'Meningioma', 'Pituitary', 'No_tumor')",
              "grade": "String (e.g., Grade IV)",
              "location": "String (The primary anatomical location, e.g. 'Right Frontal Lobe')",
              "estimated_depth_mm": "Float (Estimate the tumor's depth from the nearest cortical surface in mm. e.g., 5.0, 15.0, 30.0)",
              "pathology_analysis": "String (A 2-3 sentence description of the tumor's visual characteristics).",
              "recommendation": "String (A brief, actionable clinical recommendation).",
              "laser_parameters": {
                "power_W": "Float",
                "energy_J": "Float",
                "duration_s": "Float",
                "target_temp_C": "Float"
              }
            }
            """

            headers = {"Authorization": f"Bearer {self.api_key}"}
            user_content = [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": prompt}]
            api_messages = [{"role": "user", "content": user_content}]
            payload = {"model": self.model, "messages": api_messages, "max_tokens": 2048}

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            # --- PARSING LOGIC ---
            raw_response = response.json()['choices'][0]['message']['content']

            json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                result_data = json.loads(json_str)

                # The rest of the text is the Markdown report
                markdown_report = raw_response.replace(json_match.group(0), "").strip()
                result_data['markdown_report'] = markdown_report

                self.result_ready.emit(result_data)
            else:
                # Fallback if AI forgets JSON: send the whole text as the report
                self.result_ready.emit({
                    "markdown_report": raw_response,
                    "classification": "AI Error",
                    "grade": "N/A",
                    "estimated_depth_mm": 15.0,
                    "pathology_analysis": "Could not parse structured data from AI.",
                    "recommendation": "Check AI prompt or model."
                })

        except Exception as e:
            self.result_ready.emit({"error": f"Cloud AI analysis failed: {str(e)}"})
            
class FollowUpChatWorker(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, history):
        super().__init__()
        self.history = history
        self.api_key = "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"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-3-haiku"

    def run(self):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are Dr. AI, an expert radiologist. Your tone is clinical and professional. Focus ONLY on medical data and surgical oncology."
            }
            api_messages = [system_prompt] + self.history

            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"model": self.model, "messages": api_messages, "max_tokens": 1024}

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            bot_reply = response.json()['choices'][0]['message']['content']
            self.response_received.emit(bot_reply)
        except requests.exceptions.RequestException as e:
            error_msg = f"API Error: {e}"
            if e.response is not None:
                error_msg += f" - {e.response.text}"
            self.error_occurred.emit(error_msg)
        except Exception as e:
            self.error_occurred.emit(f"An unexpected error occurred: {str(e)}")