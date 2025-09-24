import dspy
import streamlit as st
import pandas as pd
import json
from dspy.teleprompt import BootstrapFewShot
import fitz  #PyMuPDF library
import os 
from dotenv import load_dotenv
from file_extract import parse_pdf_text
from training_data import train_data

# --- [STEP 1: CONFIGURE LANGUAGE MODELS] ---
load_dotenv()

gemini_model = dspy.LM(
    model="gemini/gemini-2.0-flash",
    provider="google",
    api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=350
)

dspy.configure(lm=gemini_model)


# ---  DEFINE SIGNATURES AND MODULE] ---



class ExtractMedicalMetadata(dspy.Signature):
    """Extract structured metadata fields from a medical document."""
    document_text = dspy.InputField(desc="The content of the medical PDF.")

    Classification = dspy.OutputField(desc="The classification of the document")
    DocumentDate = dspy.OutputField(desc="The date the document was created or signed")
    LabName = dspy.OutputField(desc="The laboratory or facility name")
    ExpirationDate = dspy.OutputField(desc="The expiration date, if available")
    ClassificationReason = dspy.OutputField(desc="The reason for classification")
    PersonnelName = dspy.OutputField(desc="The name of the personnel mentioned")
    OrganizationName = dspy.OutputField(desc="The name of the organization involved")
    StudyName = dspy.OutputField(desc="The study name related to the document")
    Country = dspy.OutputField(desc="The country where the study or lab is located")
    SiteNumber = dspy.OutputField(desc="The site number, if available")

class MedicalQASystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extracter = dspy.ChainOfThought(ExtractMedicalMetadata)

    def forward(self, document_text):
        metadata = self.extracter(document_text=document_text)
        return dspy.Prediction(
            Classification=metadata.Classification,
            DocumentDate=metadata.DocumentDate,
            LabName=metadata.LabName,
            ExpirationDate=metadata.ExpirationDate,
            ClassificationReason=metadata.ClassificationReason,
            PersonnelName=metadata.PersonnelName,
            OrganizationName=metadata.OrganizationName,
            StudyName=metadata.StudyName,
            Country=metadata.Country,
            SiteNumber=metadata.SiteNumber,
        )


# This step fine-tunes the prompts for your program.
qa_program = MedicalQASystem()
def simple_accuracy(gold, pred,trace=None):

    correct = sum(getattr(gold, f) == getattr(pred, f)
        for f in [
            "Classification","DocumentDate","LabName","ExpirationDate",
            "ClassificationReason","PersonnelName","OrganizationName",
            "StudyName","Country","SiteNumber"
        ]
    )
    return correct / 10.0
teleprompter = BootstrapFewShot(metric=simple_accuracy, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized_qa_program = teleprompter.compile(student=qa_program, trainset=train_data)


# ---  USE THE FINE-TUNED MODEL WITH A PDF ---

# #  Define PDF Path and your question
pdf_file_path = "Trainset\SubI_CV_Byron_Beer_1005.pdf"  # Make sure this file exists in your folder!
user_question = "What medication was prescribed to the patient?"


# Parse the PDF to get its text content
document_content = parse_pdf_text(pdf_file_path)

if document_content:

# Assume `prediction` is a dspy.Prediction
    prediction= optimized_qa_program(document_text=document_content)
    result_dict = {
        "Classification": prediction.Classification,
        "DocumentDate": prediction.DocumentDate,
        "LabName": prediction.LabName,
        "ExpirationDate": prediction.ExpirationDate,
        "ClassificationReason": prediction.ClassificationReason,
        "PersonnelName": prediction.PersonnelName,
        "OrganizationName": prediction.OrganizationName,
        "StudyName": prediction.StudyName,
        "Country": prediction.Country,
        "SiteNumber": prediction.SiteNumber,
    }

    # Save to JSON file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    # Inspect the last prompt sent to the Gemini model
    gemini_model.inspect_history(n=1)
else:
    print(f"Could not process the PDF file.")