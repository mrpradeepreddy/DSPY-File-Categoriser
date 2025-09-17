import dspy
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


# --- [STEP 2: DEFINE SIGNATURES AND MODULE] ---

medical_fields = ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Dermatology"]

class CategorizeMedicalDoc(dspy.Signature):
    """Categorizes the text from a medical document into a specific medical field."""
    document_text = dspy.InputField(desc="The content of the medical PDF.")
    medical_field = dspy.OutputField(desc=f"Choose from: {', '.join(medical_fields)}")

class MedicalQASystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.categorizer = dspy.ChainOfThought(CategorizeMedicalDoc)

    def forward(self, document_text):
        category_prediction = self.categorizer(document_text=document_text)
        return dspy.Prediction(
            medical_field=category_prediction.medical_field,
        )


# --- [STEP 4: COMPILE THE PROGRAM] ---
# This step fine-tunes the prompts for your program.
qa_program = MedicalQASystem()

teleprompter = BootstrapFewShot(metric=None, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized_qa_program = teleprompter.compile(student=qa_program, trainset=train_data)


# --- [STEP 5: USE THE FINE-TUNED MODEL WITH A PDF] ---

# 1. Define PDF Path and your question
pdf_file_path = "Testset\onco_test_1.pdf"  # Make sure this file exists in your folder!
user_question = "What medication was prescribed to the patient?"

# 2. Parse the PDF to get its text content
document_content = parse_pdf_text(pdf_file_path)

if document_content:
    # 3. Run the optimized program with the extracted text
    prediction = optimized_qa_program(document_text=document_content)

    print(f"Analyzing Document: '{pdf_file_path}'")
    print("--- Model Output ---")
    print(f"Predicted Medical Field: {prediction.medical_field}")

    # Inspect the last prompt sent to the Gemini model
    gemini_model.inspect_history(n=1)
else:
    print(f"Could not process the PDF file.")