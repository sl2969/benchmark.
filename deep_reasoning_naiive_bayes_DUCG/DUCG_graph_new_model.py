import json
import numpy as np
import pandas as pd

##############################################################################
# 1) LOAD THE CSV AND TRANSCRIPT
##############################################################################

def load_weights(csv_path: str) -> pd.DataFrame:
    """
    Loads the DUCG weight table, with columns like:
      [ 'Parameter','CHF','Cardiac Tamponade','Pulmonary Embolism',
        'Pleural Effusion','Myocardial Infarction','Aortic Aneurysm/Dissection','Benign' ]
    We set the 'Parameter' column as the DataFrame index.
    """
    df = pd.read_csv(csv_path)
    if "Parameter" not in df.columns:
        raise ValueError("CSV must have 'Parameter' as a column.")
    df = df.set_index("Parameter")
    return df

def load_transcript(json_path: str) -> dict:
    """
    Loads the patient transcript JSON from the given path.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

##############################################################################
# 2) PARSE THE TRANSCRIPT INTO A SIMPLE EVIDENCE DICT
##############################################################################

def parse_evidence_from_transcript(transcript: dict) -> dict:
    """
    Flatten the JSON structure into a dict: variable_name -> True/False/None
    True means present, False means absent, None means unknown/no data.
    """
    evidence = {}
    
    def interpret_value(val):
        """
        Attempt to interpret if 'val' indicates presence or absence.
        Return True (present), False (absent), None (unknown).
        """
        if isinstance(val, (int, float)):
            # If severity scale 1..10 => present if >0
            return True if val > 0 else False
        elif isinstance(val, str):
            val_l = val.lower().strip()
            if val_l == "yes":
                return True
            elif val_l == "no":
                return False
            elif val_l == "unknown":
                return None

            # For words like 'acute', 'chronic' => assume present
            if val_l in ["acute", "chronic"]:
                return True

            # If echocardiogram specifically says 'tamponade', treat that as True
            if val_l == "tamponade":
                return True

            # If the variable is some other string => uncertain
            return None
        return None

    # Now fill out evidence for each section
    # Patient Demographics
    for k, v in transcript.get("Patient Demographics", {}).items():
        evidence[k] = interpret_value(v)
    
    # Medical Background
    for k, v in transcript.get("Medical Background", {}).items():
        evidence[k] = interpret_value(v)

    # Symptoms & Severity
    for k, v in transcript.get("Symptoms & Severity", {}).items():
        evidence[k] = interpret_value(v)
    
    # Symptom Onset
    onset_dict = transcript.get("Symptom Onset and Duration", {})
    for symptom, onset_val in onset_dict.items():
        onset_key = f"{symptom}_onset"
        evidence[onset_key] = interpret_value(onset_val)
    
    # Vital Signs
    for k, v in transcript.get("Vital Signs", {}).items():
        evidence[k] = interpret_value(v)

    # Contextual Triggers
    for k, v in transcript.get("Contextual Triggers", {}).items():
        evidence[k] = interpret_value(v)

    # Laboratory Tests
    labs = transcript.get("Laboratory Tests", {})
    for lab_name, lab_info in labs.items():
        lab_val = lab_info.get("value", "unknown")
        evidence[lab_name] = interpret_value(lab_val)

    # Imaging Studies
    imaging = transcript.get("Imaging Studies", {})
    for k, v in imaging.items():
        evidence[k] = interpret_value(v)

    # Pain Characteristics
    pains = transcript.get("Pain Characteristics", {})
    for symp_name, details in pains.items():
        if isinstance(details, dict):
            for subk, subv in details.items():
                param_key = f"{symp_name}_{subk}"
                evidence[param_key] = interpret_value(subv)
        else:
            evidence[symp_name] = interpret_value(details)

    return evidence

##############################################################################
# 3) NAIVE BAYES INFERENCE WITH RED-FLAG OVERRIDE
##############################################################################

def compute_naive_bayes(weights_df: pd.DataFrame, evidence: dict):
    """
    Interpret each w_{v,d} as P(var present | disease).
    If var is present => multiply by w_{v,d}.
    If var is absent  => multiply by (1 - w_{v,d}).
    If var is unknown => skip.

    Then we do a post-hoc "red flag" override:
    If echocardiogram specifically shows 'tamponade', we multiply
    Cardiac Tamponade's likelihood by 10.
    """
    diseases = weights_df.columns.tolist()

    # 1) Simple uniform priors
    priors = {d: 1 / len(diseases) for d in diseases}

    # 2) Accumulate log probabilities
    logp = {d: np.log(priors[d]) for d in diseases}

    # Naïve Bayes loop
    for var, val in evidence.items():
        if var not in weights_df.index:
            continue
        if val is None:
            continue

        p_row = weights_df.loc[var]
        for d in diseases:
            p_present = p_row[d]
            p_present = max(min(p_present, 0.999999), 1e-7)

            if val is True:
                # multiply by p_present => add log(p_present)
                logp[d] += np.log(p_present)
            else:
                # val is False => multiply by (1 - p_present)
                p_not = max(min(1.0 - p_present, 0.999999), 1e-7)
                logp[d] += np.log(p_not)

    # === RED FLAG / OVERRIDE for Tamponade ===
    # If echocardiogram is 'tamponade' => multiply Tamponade's likelihood by 10
    # i.e. add ln(10) ~ +2.302585
    # You could add synergy checks too.
    if evidence.get("echocardiogram") is True:
        logp["Cardiac Tamponade"] += np.log(10)

    # Convert logp -> unnormalized likelihood
    raw_likelihood = {d: np.exp(lp) for d, lp in logp.items()}

    # Convert to posterior by normalizing
    max_lp = max(logp.values())
    exps = {d: np.exp(logp[d] - max_lp) for d in diseases}
    denom = sum(exps.values())
    posterior = {d: exps[d] / denom for d in diseases}

    return posterior, logp, raw_likelihood

def log_to_likelihood(logp: dict) -> dict:
    """
    Converts a dictionary of log-likelihoods to normalized likelihoods.
    Args:
        logp: A dictionary where keys are diseases and values are log-likelihoods.
    Returns:
        A dictionary of normalized likelihoods.
    """
    # Exponentiate the log-probabilities to get unnormalized likelihoods
    unnormalized_likelihoods = {d: np.exp(lp) for d, lp in logp.items()}
    
    # Normalize so that the likelihoods sum to 1
    total = sum(unnormalized_likelihoods.values())
    normalized_likelihoods = {d: likelihood / total for d, likelihood in unnormalized_likelihoods.items()}
    
    return normalized_likelihoods

##############################################################################
# 4) MAIN
##############################################################################

if __name__ == "__main__":
    # Adjust these as needed
    WEIGHTS_CSV = "DUCG_Weight_Table.csv"
    TRANSCRIPT_JSON = "transcript_output.json"

    # 1) Load Weights
    weights_df = load_weights(WEIGHTS_CSV)

    # 2) Load Transcript
    transcript = load_transcript(TRANSCRIPT_JSON)

    # 3) Parse Evidence
    evidence_dict = parse_evidence_from_transcript(transcript)

    # 4) Run Naïve Bayes
    posterior, logp, raw_like = compute_naive_bayes(weights_df, evidence_dict)

    # Convert log-likelihoods to readable likelihoods
    readable_likelihoods = log_to_likelihood(logp)

    # 5) Print Results
    print("==== Posterior probabilities (normalized) ====")
    for disease, prob in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
        print(f"{disease:30s} => {prob:.3f}")

    print("\n==== Unnormalized Likelihood (scientific notation) ====")
    for disease, val in sorted(raw_like.items(), key=lambda x: x[1], reverse=True):
        print(f"{disease:30s} => {val:.6e}")

    print("\n==== Final Log Probability ====")
    for disease, lp_val in sorted(logp.items(), key=lambda x: x[1], reverse=True):
        print(f"{disease:30s} => {lp_val:.3f}")

    print("\n==== Final Likelihoods (Normalized) ====")
    for disease, likelihood in sorted(readable_likelihoods.items(), key=lambda x: x[1], reverse=True):
        print(f"{disease:30s} => {likelihood:.6f}")