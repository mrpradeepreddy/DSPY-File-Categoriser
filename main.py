import dspy
import streamlit as st
import pandas as pd
import json
from dspy.teleprompt import BootstrapFewShot
import fitz  # PyMuPDF library
import os 
from dotenv import load_dotenv
from file_extract import parse_pdf_text
from training_data import train_data

# --- [STEP 1: CONFIGURE LANGUAGE MODELS] ---
load_dotenv()


if "dspy_configured" not in st.session_state:    
    gemini_model = dspy.LM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=os.getenv("GEMINI_API_KEY"),
        max_output_tokens=350
    )

    dspy.configure(lm=gemini_model)
    st.session_state["dspy_configured"] = True

# ---  DEFINE SIGNATURES AND MODULE] ---
class ExtractMedicalMetadata(dspy.Signature):
    """Extract structured metadata fields from a medical document."""
    document_text = dspy.InputField(desc="The content of the medical PDF.")

    Classification = dspy.OutputField(desc="The classification of the document like Principal Investigator Curriculum Vitae ,Sub-Investigator Curriculum Vitae ,Site Training Material ,Site Evidence of Training ,if the value doesnot exist then return N/A")
    DocumentDate = dspy.OutputField(desc="The date the document was created or signed like mm-dd-yyyy. if the value doesnot exist then return N/A") 
    LabName = dspy.OutputField(desc="Extract the name of the laboratory associated with the site as specified in the documment like Site Name: Kevinstad Memorial Clinic.Display the extracted lab name as Lab Name in the output. If no lab name is mentioned or identifiable, leave the field blank. ")
    ExpirationDate = dspy.OutputField(desc="The expiration date, if available like mm-dd-yyyy. If no expiration date is mentioned or identifiable, leave the field blank or return N/A.")
    ClassificationReason = dspy.OutputField(desc="The reason for classification like Syneos Health")
    PersonalName = dspy.OutputField(desc="Extract the name of the person mentioned in the text.That may be indicated as the Personal name=Dr. John Doe or Investigator Name: John Doe. If no personal name is mentioned or identifiable, leave the field blank.")
    OrganizationName = dspy.OutputField(desc="The name of the organization involved .If no organization name is mentioned or identifiable, leave the field blank or return N/A.")
    StudyName = dspy.OutputField(desc="The study name related to the document")
    Country = dspy.OutputField(desc="The country where the study or lab is located like United States,Canada,India or return N/A if not available")
    SiteNumber = dspy.OutputField(desc="The site number, if available like  1007 or  Site Number: 1007")
    Type=dspy.OutputField(desc="The type of document like Site Management,if there is no type then return N/A")
    Subtype=dspy.OutputField(desc="The subtype of document like Site Set-up,Site Initiation,General,Site Selection,Site Management.If there is no subtype then return N/A")

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
            PersonalName=metadata.PersonalName,
            OrganizationName=metadata.OrganizationName,
            StudyName=metadata.StudyName,
            Country=metadata.Country,
            SiteNumber=metadata.SiteNumber,
            Type=metadata.Type,
            Subtype=metadata.Subtype
        )


# Fine-tune prompts
qa_program = MedicalQASystem()
def simple_accuracy(gold, pred,trace=None):
    correct = sum(getattr(gold, f) == getattr(pred, f)
        for f in [
            "Classification","DocumentDate","LabName","ExpirationDate",
            "ClassificationReason","PersonalName","OrganizationName",
            "StudyName","Country","SiteNumber","Type","Subtype"
        ]
    )
    return correct / 10.0

teleprompter = BootstrapFewShot(metric=simple_accuracy, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized_qa_program = teleprompter.compile(student=qa_program, trainset=train_data)

# --- STREAMLIT UI ---
st.title("📑 Medical Document Metadata Extractor")

uploaded_file = st.file_uploader("Upload a medical PDF", type=["pdf", "docx", "odt", "txt", "rtf", "xlsx", "csv", "pptx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_path = os.path.join("temp_files", file_name)
    os.makedirs("temp_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    document_content = parse_pdf_text(file_path)

    if document_content:
        prediction = optimized_qa_program(document_text=document_content)

        result_dict = {
            "Classification": prediction.Classification,
            "DocumentDate": prediction.DocumentDate,
            "LabName": prediction.LabName,
            "ExpirationDate": prediction.ExpirationDate,
            "ClassificationReason": prediction.ClassificationReason,
            "PersonalName": prediction.PersonalName,
            "OrganizationName": prediction.OrganizationName,
            "StudyName": prediction.StudyName,
            "Country": prediction.Country,
            "SiteNumber": prediction.SiteNumber,
            "Type": prediction.Type,
            "Subtype": prediction.Subtype
        }

        st.subheader("Extracted Metadata")
        st.json(result_dict)

        # Save JSON file
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        st.success("Metadata saved to output.json ✅")
    else:
        st.error("Could not process the PDF file.")
