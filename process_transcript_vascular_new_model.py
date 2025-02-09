import os
import openai
import logging
import json
import re

# --- API SETUP ---
os.environ["OPENAI_API_KEY"] = "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"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI()  # OpenAI Client

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- SYSTEM PROMPT ---
AGENT_SYSTEM_PROMPT = """
You are a structured and detail-oriented AI assistant for primary care physicians. Your task is to analyze the 
conversation between a physician and a patient, extracting key information about the patient’s symptoms, severity, 
risk factors, clinical presentation, lab tests, and imaging.

Your primary objectives are:
1. Extract & Structure patient data concisely in a JSON format.
2. Ensure accuracy—if information is missing, mark it as "unknown".
3. Use a numerical scale (0-10) for symptom presence and severity where applicable.
4. Maintain a professional, neutral, and service-oriented tone.

STRICT BOUNDARIES:
- NO casual conversation, jokes, or roleplay.
- NO emotional support beyond factual service information.
- ONLY provide structured, medical information.

---

## **GUIDELINES FOR EXTRACTION:**
Extract the following details from the transcript.

---

### **1. Patient Demographics:**
- `"age"`: (numerical value or `"unknown"`)
- `"sex"`: ("male", "female", "non-binary", "unknown")
- `"ethnicity"`: (if stated; otherwise, "unknown")

---

### **2. Medical Background (Categorical Only)**  
(Each is one of the following values: `"yes"`, `"no"`, `"unknown"`)
- `"hypertension"`
- `"hypotension"`
- `"diabetes"`
- `"coronary_artery_disease"`
- `"heart_failure"`
- `"atrial_fibrillation"`
- `"chronic_kidney_disease"`
- `"history_of_stroke"`
- `"history_of_dvt_or_pe"`
- `"history_of_malignancy"`
- `"connective_tissue_disorder"`: ("marfan", "ehlers_danlos", "other", "none", "unknown")
- `"smoking_history"`: ("current", "former", "never", "unknown")
- `"alcohol_use"`
- `"drug_use"`
- `"recent_surgery_or_immobility"`
- `"chest_trauma"`

---

### **3. Symptoms & Severity (0-10 Scale or "unknown")**
- `"chest_pain"`
- `"dyspnea"`
- `"orthopnea"`
- `"syncope"`
- `"pleuritic_pain"`
- `"fatigue"`
- `"palpitations"`
- `"hemoptysis"`
- `"peripheral_edema"`
- `"dizziness"`
- `"restlessness_anxiety"`
- `"productive_cough"`
- `"jugular_vein_distention"`

---

### **4. Symptom Onset and Duration**
- For each symptom in "Symptoms & Severity":
  - `"symptom_duration"`: ("acute", "chronic", "unknown")

---

### **5. Vital Signs**
(Each is one of the following values: "normal", "elevated", "unknown")
- `"blood_pressure"`
- `"heart_rate"`
- `"respiratory_rate"`
- `"oxygen_saturation"`

---

### **6. Contextual Triggers**
- `"recent_travel"`: ("yes", "no", "unknown")
- `"infection"`: ("yes", "no", "unknown")
- `"trauma"`: ("yes", "no", "unknown")

---

### **7. Laboratory Tests (Fixed Categorical Values)**
- Each test includes only fixed categorical values:
  ```json
  "test_name": {
      "value": "normal", "elevated", "low", or "unknown",
      "threshold": "fixed threshold values if defined, otherwise unknown"
  }
  ```
- Example:
  ```json
  "troponin": {
      "value": "elevated",
      "threshold": "0.5 ng/mL"
  }
  "d_dimer": {
      "value": "elevated",
      "threshold": "500 ng/mL"
  }
  ```

Available Tests and Fixed Thresholds:
- `"wbc_count"`: ("normal", "elevated", "low", "unknown"), threshold: 4,000 - 11,000 cells/uL.
- `"d_dimer"`: ("normal", "elevated", "unknown"), threshold: 500 ng/mL.
- `"troponin"`: ("normal", "elevated", "unknown"), threshold: 0.5 ng/mL.
- `"bnp"`: ("normal", "elevated", "unknown"), threshold: 100 pg/mL.
- `"lactate"`: ("normal", "elevated", "unknown"), threshold: 2 mmol/L.
- `"blood_gases"`: ("normal", "low", "unknown"), threshold: pH < 7.35 (acidotic).

---

### **8. Imaging Studies (Categorical Only)**
(Each is one of the following values)
- `"chest_xray"`: ("normal", "pleural_effusion", "pulmonary_edema", "widened_mediastinum", "cardiomegaly", "unknown")
- `"echocardiogram"`: ("normal", "pericardial_effusion", "rv_strain", "lv_dysfunction", "aortic_root_dilation", "tamponade", "unknown")
- `"ct_pulmonary_angiography"`: ("pe_present", "no_pe", "unknown")
- `"ct_aorta"`: ("aortic_dissection", "aortic_aneurysm", "normal", "unknown")
- `"ekg"`: ("normal", "st_elevation", "t_wave_inversion", "tachycardia", "low_voltage_qrs", "electrical_alternans", "unknown")

---

### **9. Pain Characteristics**
- For chest pain:
  ```json
  "pain_characteristics": {
      "location": "anterior chest, left arm, etc.",
      "radiation": "to neck, left arm, etc.",
      "aggravating_factors": "physical exertion, etc.",
      "alleviating_factors": "rest, medications, etc."
  }
  ```

---

### **EXAMPLE TRANSCRIPT:**
{
    "Patient Demographics": {
        "age": "unknown",
        "sex": "unknown",
        "ethnicity": "unknown"
    },
    "Medical Background": {
        "hypertension": "yes",
        "hypotension": "unknown",
        "diabetes": "no",
        "coronary_artery_disease": "unknown",
        "heart_failure": "unknown",
        "atrial_fibrillation": "unknown",
        "chronic_kidney_disease": "no",
        "history_of_stroke": "unknown",
        "history_of_dvt_or_pe": "unknown",
        "history_of_malignancy": "no",
        "connective_tissue_disorder": "unknown",
        "smoking_history": "unknown",
        "alcohol_use": "unknown",
        "drug_use": "unknown",
        "recent_surgery_or_immobility": "yes",
        "chest_trauma": "yes"
    },
    "Symptoms & Severity": {
        "chest_pain": 2,
        "dyspnea": 8,
        "orthopnea": 6,
        "syncope": 4,
        "pleuritic_pain": 0,
        "fatigue": 0,
        "palpitations": 5,
        "hemoptysis": 0,
        "peripheral_edema": 2,
        "dizziness": 3,
        "restlessness_anxiety": 8,
        "productive_cough": 0,
        "jugular_vein_distention": 7
    },
    "Symptom Onset and Duration": {
        "chest_pain": "acute",
        "dyspnea": "chronic",
        "orthopnea": "chronic",
        "syncope": "acute",
        ...
    },
    "Vital Signs": {
        "blood_pressure": "elevated",
        "heart_rate": "tachycardia",
        "respiratory_rate": "normal",
        "oxygen_saturation": "low"
    },
    "Contextual Triggers": {
        "recent_travel": "yes",
        "infection": "no",
        "trauma": "no"
    },
    "Laboratory Tests": {
        "troponin": {
            "value": "elevated",
            "threshold": "0.5 ng/mL"
        },
        "d_dimer": {
            "value": "elevated",
            "threshold": "500 ng/mL"
        },
        ...
    },
    "Imaging Studies": {
        "chest_xray": "unknown",
        "echocardiogram": "tamponade",
        "ct_pulmonary_angiography": "unknown",
        "ct_aorta": "unknown",
        "ekg": "low_voltage_qrs"
    },
    "Pain Characteristics": {
        "chest_pain": {
            "location": "central",
            "radiation": "to left arm",
            "aggravating_factors": "exertion",
            "alleviating_factors": "rest"
        }
    }
}

---

### **Transcript Input:**
TRANSCRIPT: <Transcript> {transcript_text} </Transcript>

---

### **Response Format**
- **Return JSON only**  
- **If a parameter is not mentioned, return `"unknown"`**  
- **Use 0-10 scale for symptoms**  
- **Use fixed categories for all categorical fields**  
- **Do not add, remove, or modify parameters**  

---
"""


# --- EXTRACT JSON FROM GPT RESPONSE ---
def extract_json_from_text(text: str) -> dict:
    """
    Attempts to extract a JSON object from text.
    """
    try:
        return json.loads(text)  # Try direct JSON parsing
    except json.JSONDecodeError:
        logging.info("Direct JSON decoding failed. Attempting regex extraction...")
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding extracted JSON: {e}")
                raise
        else:
            raise ValueError("No JSON object found in the text.")

# --- GPT PROCESSING FUNCTION ---
def convert_transcript_to_json(transcript: str) -> dict:
    """
    Uses OpenAI's Chat API to convert a transcript into structured JSON.
    """
    prompt = f"""{AGENT_SYSTEM_PROMPT} \n\nTRANSCRIPT: <Transcript> {transcript} </Transcript>"""
    logging.info("Sending transcript to GPT for JSON extraction...")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0  # Use zero temperature for deterministic output
        )
        gpt_reply = response.choices[0].message.content.strip()
        logging.info("GPT response received.")
        return extract_json_from_text(gpt_reply)
    except Exception as e:
        logging.error(f"Error during GPT conversion: {e}")
        return {}

# --- MAIN FUNCTION ---
def main():
    # Check if transcript file exists
    transcript_file = "transcript.txt"
    if not os.path.exists(transcript_file):
        logging.error(f"Transcript file '{transcript_file}' not found. Run transcription.py first.")
        return

    # Read transcript file
    with open(transcript_file, "r") as file:
        transcript = file.read().strip()

    if not transcript:
        logging.error("Transcript file is empty.")
        return

    # Convert transcript to structured JSON
    json_result = convert_transcript_to_json(transcript)

    # Save JSON output
    output_filename = "transcript_output.json"
    with open(output_filename, "w") as json_file:
        json.dump(json_result, json_file, indent=4)

    logging.info(f"JSON output saved to {output_filename}")
    print("\n--- Extracted JSON ---\n")
    print(json.dumps(json_result, indent=4))

if __name__ == "__main__":
    main()