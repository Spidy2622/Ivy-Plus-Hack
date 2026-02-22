"""
HemoSense — CCHF Risk & Precaution Report Generator

Generates structured clinical decision-support reports and PDF exports
based on existing prediction outputs. No ML logic is performed here.
"""

import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


# ── Feature display-name mapping ──────────────────────────────────────────────
FEATURE_LABELS = {
    "platelet_low": "Low Platelet Count (Thrombocytopenia)",
    "region_risk": "Geographic Region Risk",
    "endemic_level": "Endemic Area Level",
    "bleeding": "Hemorrhagic Bleeding",
    "occupation_risk": "High-Risk Occupation",
    "shock_signs": "Signs of Shock",
    "symptom_days": "Duration of Symptoms",
    "fever": "Fever",
    "liver_impairment": "Liver Impairment / Failure",
    "livestock_contact": "Livestock Contact",
    "slaughter_exposure": "Slaughterhouse Exposure",
    "healthcare_exposure": "Healthcare Setting Exposure",
    "days_since_tick": "Recent Tick Bite (days)",
    "tick_bite": "Tick Bite History",
    "wbc_low": "Leukopenia (Low WBC)",
    "ast_alt_high": "Elevated Liver Enzymes (AST/ALT)",
    "headache": "Headache",
    "neck_pain": "Neck Pain",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "dizziness": "Dizziness",
    "muscle_pain": "Muscle Pain",
    "abdominal_pain": "Abdominal Pain",
    "photophobia": "Photophobia",
    "human_contact": "Contact with CCHF Patient",
    "month_sin": "Seasonal Factor",
    "month_cos": "Seasonal Factor",
    "days_since_contact": "Days Since Contact Exposure",
}

# ── Risk-level → interpretation text ──────────────────────────────────────────
RISK_INTERPRETATIONS = {
    "Low": (
        "The model indicates a low probability of severe CCHF based on the "
        "provided clinical and exposure data. The patient's symptom profile and "
        "exposure history do not strongly align with high-risk patterns. However, "
        "clinical vigilance should be maintained as CCHF can progress rapidly."
    ),
    "Medium": (
        "The model indicates a moderate probability of CCHF. One or more risk "
        "factors — such as geographic exposure, tick contact, or early hemorrhagic "
        "signs — are present. Further clinical evaluation, laboratory confirmation, "
        "and close monitoring are recommended."
    ),
    "High": (
        "The model indicates a high probability of severe CCHF. Multiple significant "
        "risk factors are present, which may include hemorrhagic symptoms, "
        "thrombocytopenia, liver involvement, or high-risk geographic/occupational "
        "exposure. Urgent clinical intervention and infection-control measures are "
        "strongly recommended."
    ),
}

# ── Risk-level → precaution lists (WHO-aligned) ──────────────────────────────
PRECAUTIONS = {
    "Low": [
        "Practice tick-bite prevention: use DEET-based repellents and wear long clothing in endemic areas",
        "Monitor for symptom onset (fever, myalgia, bleeding) for 14 days after potential exposure",
        "Maintain standard hygiene precautions when handling animals or animal products",
        "Educate household members about CCHF transmission routes",
        "Schedule follow-up if symptoms develop",
    ],
    "Medium": [
        "Seek medical evaluation promptly for laboratory confirmation (CBC, liver function, CCHF serology)",
        "Implement contact precautions and respiratory hygiene",
        "Isolate from other patients pending test results",
        "Avoid invasive procedures until bleeding risk is assessed",
        "Notify local public-health authorities of suspected case",
        "Begin supportive care: IV fluids, antipyretics (avoid NSAIDs)",
    ],
    "High": [
        "Urgent hospitalization in an isolation facility with infection-control capacity",
        "Implement strict contact and droplet precautions per WHO IPC guidelines",
        "Request PCR-based CCHF confirmation testing immediately",
        "Initiate supportive intensive care: platelet transfusion, FFP, packed RBCs as needed",
        "Consider Ribavirin therapy per WHO guidance if within early disease window",
        "Use full PPE (gown, gloves, N95, face shield) for all patient interactions",
        "Handle all specimens as Biosafety Level 4 material",
        "Restrict and log all contacts for 14-day surveillance",
    ],
}

# ── Medical guidance by risk level ────────────────────────────────────────────
MEDICAL_GUIDANCE = {
    "Low": (
        "Seek medical care if you develop fever, unexplained bleeding, severe headache, "
        "or muscle pain within 14 days of potential exposure. Inform your healthcare provider "
        "about any tick bites or contact with livestock in endemic areas."
    ),
    "Medium": (
        "Seek medical evaluation within 24 hours. Request a complete blood count and liver "
        "function tests. If symptoms worsen — especially bleeding, confusion, or high fever — "
        "proceed to the nearest facility with isolation capability immediately."
    ),
    "High": (
        "Seek emergency medical care immediately. This is a medical urgency. Present to "
        "the nearest hospital with infection-control and intensive-care capability. "
        "Inform healthcare staff of suspected CCHF before arrival so appropriate "
        "precautions can be prepared."
    ),
}

DISCLAIMER = (
    "⚠️ This report is an AI-generated clinical decision-support summary based on "
    "WHO-aligned data patterns. It does not constitute a medical diagnosis, replace "
    "professional clinical judgment, or substitute laboratory-confirmed testing. "
    "Always consult a qualified healthcare professional for diagnosis and treatment."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_cchf_report(
    inputs: dict,
    prediction: str,
    stage: str,
    risk_probabilities: dict,
    stage_probabilities: dict,
    feature_importance: list | None,
) -> dict:
    """Return a structured dict of report sections.

    Parameters
    ----------
    inputs : dict
        Patient-entered data (symptom flags, exposure flags, lab values, region, etc.)
    prediction : str
        Predicted risk label ("Low", "Medium", or "High").
    stage : str
        Predicted disease stage.
    risk_probabilities : dict
        {class_label: probability} from the risk model.
    stage_probabilities : dict
        {class_label: probability} from the stage model.
    feature_importance : list | None
        List of dicts [{"feature": ..., "importance": ...}, ...] from feature_importance.json.
    """

    risk_level = prediction  # "Low", "Medium", or "High"

    # ── Key risk factors ─────────────────────────────────────────────────
    key_factors = _extract_key_factors(inputs, feature_importance)

    # ── Patient summary ──────────────────────────────────────────────────
    active_symptoms = [s for s in [
        "fever", "bleeding", "headache", "muscle_pain", "vomiting",
        "dizziness", "neck_pain", "photophobia", "abdominal_pain", "diarrhea"
    ] if inputs.get(s)]

    exposure_flags = [e for e in [
        "tick_bite", "livestock_contact", "slaughter_exposure",
        "healthcare_exposure", "human_contact"
    ] if inputs.get(e)]

    patient_summary = (
        f"Patient presents with {len(active_symptoms)} symptom(s) "
        f"({', '.join(active_symptoms) if active_symptoms else 'none reported'}) "
        f"and {len(exposure_flags)} exposure factor(s) "
        f"({', '.join(e.replace('_', ' ').title() for e in exposure_flags) if exposure_flags else 'none reported'}). "
        f"Region: {inputs.get('region', 'N/A')}. "
        f"Occupation: {inputs.get('occupation', 'N/A')}. "
        f"Month of onset: {inputs.get('month', 'N/A')}."
    )

    # ── Confidence string ────────────────────────────────────────────────
    risk_conf = max(risk_probabilities.values()) * 100
    confidence_text = (
        f"The model predicts **{risk_level.upper()}** risk with "
        f"**{risk_conf:.1f}%** confidence. "
        f"Predicted disease stage: **{str(stage).capitalize()}** "
        f"({max(stage_probabilities.values()) * 100:.1f}% confidence)."
    )

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "patient_summary": patient_summary,
        "risk_level": risk_level,
        "confidence_text": confidence_text,
        "key_factors": key_factors,
        "interpretation": RISK_INTERPRETATIONS.get(risk_level, RISK_INTERPRETATIONS["Medium"]),
        "precautions": PRECAUTIONS.get(risk_level, PRECAUTIONS["Medium"]),
        "medical_guidance": MEDICAL_GUIDANCE.get(risk_level, MEDICAL_GUIDANCE["Medium"]),
        "disclaimer": DISCLAIMER,
    }


def _extract_key_factors(inputs: dict, feature_importance: list | None) -> list[dict]:
    """Identify the top contributing risk factors for this patient.

    Combines feature importance ranking with the patient's actual positive /
    elevated values to surface only factors that are *both* important to the
    model *and* present in this patient.
    """
    if not feature_importance:
        return []

    # Binary / flag features that count as "active" when truthy
    binary_features = {
        "fever", "bleeding", "headache", "muscle_pain", "vomiting",
        "dizziness", "neck_pain", "photophobia", "abdominal_pain", "diarrhea",
        "tick_bite", "livestock_contact", "slaughter_exposure",
        "healthcare_exposure", "human_contact",
        "platelet_low", "wbc_low", "ast_alt_high", "liver_impairment", "shock_signs",
    }

    # Continuous features that count as "active" when > 0
    continuous_features = {
        "symptom_days", "days_since_tick", "days_since_contact",
    }

    # Region / occupation always contribute — mark as active when elevated
    always_active = {"region_risk", "endemic_level", "occupation_risk", "month_sin", "month_cos"}

    key = []
    for entry in feature_importance:
        feat = entry["feature"]
        imp = entry["importance"]
        if imp < 0.001:
            continue  # skip negligible features

        active = False
        if feat in binary_features and inputs.get(feat):
            active = True
        elif feat in continuous_features and inputs.get(feat, 0) > 0:
            active = True
        elif feat in always_active:
            active = True

        if active:
            key.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "importance": imp,
            })

    # Return top 8 at most
    return key[:8]


# ══════════════════════════════════════════════════════════════════════════════
#  PDF generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report_pdf(report: dict) -> bytes:
    """Create a formatted PDF from a report dict and return raw bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=HexColor("#1e293b"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        borderPadding=(0, 0, 2, 0),
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        textColor=HexColor("#334155"),
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=20,
        bulletIndent=8,
        spaceBefore=2,
        spaceAfter=2,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8,
        textColor=HexColor("#64748b"),
        spaceBefore=16,
        leading=11,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=body_style,
        fontSize=9,
        textColor=HexColor("#64748b"),
    )

    elements = []

    # Title
    elements.append(Paragraph("HemoSense — CCHF Risk & Precaution Report", title_style))
    elements.append(Paragraph(f"Generated: {report['timestamp']}", meta_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cbd5e1")))
    elements.append(Spacer(1, 6))

    # Risk level banner
    risk_color = {"Low": "#065f46", "Medium": "#92400e", "High": "#991b1b"}.get(
        report["risk_level"], "#334155"
    )
    elements.append(
        Paragraph(
            f'<font color="{risk_color}" size="14"><b>Risk Level: {report["risk_level"].upper()}</b></font>',
            body_style,
        )
    )
    elements.append(
        Paragraph(report["confidence_text"].replace("**", ""), body_style)
    )
    elements.append(Spacer(1, 4))

    # Patient Summary
    elements.append(Paragraph("Patient Summary", heading_style))
    elements.append(Paragraph(report["patient_summary"], body_style))

    # Key Risk Factors
    elements.append(Paragraph("Key Risk Factors", heading_style))
    if report["key_factors"]:
        table_data = [["#", "Factor", "Model Importance"]]
        for i, f in enumerate(report["key_factors"], 1):
            table_data.append([str(i), f["label"], f"{f['importance']:.4f}"])
        t = Table(table_data, colWidths=[0.4 * inch, 3.8 * inch, 1.4 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("No dominant risk factors identified from model importance data.", body_style))

    # Interpretation
    elements.append(Paragraph("Risk Interpretation", heading_style))
    elements.append(Paragraph(report["interpretation"], body_style))

    # Precautions
    elements.append(Paragraph("Recommended Precautions", heading_style))
    for p in report["precautions"]:
        elements.append(Paragraph(f"• {p}", bullet_style))

    # Medical Guidance
    elements.append(Paragraph("Medical Guidance", heading_style))
    elements.append(Paragraph(report["medical_guidance"], body_style))

    # Disclaimer
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#e2e8f0")))
    elements.append(Paragraph(report["disclaimer"], disclaimer_style))

    doc.build(elements)
    return buf.getvalue()
