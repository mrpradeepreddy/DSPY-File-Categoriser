import dspy
from dspy.teleprompt import BootstrapFewShot
import fitz  # PyMuPDF library
import os 

# --- NEW: PDF Parsing Function ---
def parse_pdf_text(file_path):
    """Extracts all text from a given PDF file."""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

# --- [STEP 1: CONFIGURE LANGUAGE MODELS] ---
# GOOGLE_API_KEY environment variable is set.

gemini_model = dspy.LM(
    model="gemini/gemini-2.0-flash",
    provider="google",
    api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=350
)

dspy.configure(lm=gemini_model)




# --- [STEP 2: DEFINE SIGNATURES AND MODULE] ---
# The signatures and module logic remain exactly the same.
medical_fields = ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Dermatology"]

class CategorizeMedicalDoc(dspy.Signature):
    """Categorizes the text from a medical document into a specific medical field."""
    document_text = dspy.InputField(desc="The content of the medical PDF.")
    medical_field = dspy.OutputField(desc=f"Choose from: {', '.join(medical_fields)}")

class AnswerFromDocument(dspy.Signature):
    """Answer a question based ONLY on the provided document context."""
    context = dspy.InputField(desc="The text from the medical document.")
    question = dspy.InputField(desc="A question about the document's content.")
    answer = dspy.OutputField(desc="A concise answer to the question based strictly on the context.")

class MedicalQASystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.categorizer = dspy.ChainOfThought(CategorizeMedicalDoc)
        self.qa_system = dspy.ChainOfThought(AnswerFromDocument)

    def forward(self, document_text, question):
        category_prediction = self.categorizer(document_text=document_text)
        answer_prediction = self.qa_system(context=document_text, question=question)
        return dspy.Prediction(
            medical_field=category_prediction.medical_field,
            answer=answer_prediction.answer
        )


# --- [STEP 3: PREPARE THE DATASET] ---
# Your training data remains the same. This is used to teach the model.
train_data = [
    dspy.Example(
        document_text="Patient presents with atrial fibrillation and hypertension. Echocardiogram shows reduced ejection fraction. Prescribed beta-blockers and anticoagulants.",
        medical_field="Cardiology",
        question="What was the patient prescribed?",
        answer="Beta-blockers and anticoagulants."
    ).with_inputs("document_text", "question"),
    dspy.Example(
        document_text="A study on the efficacy of immunotherapy for melanoma. Results indicate a significant improvement in progression-free survival for patients treated with PD-1 inhibitors.",
        medical_field="Oncology",
        question="What type of cancer was studied?",
        answer="Melanoma."
    ).with_inputs("document_text", "question"),
]
trainset = train_data


# --- [STEP 4: COMPILE THE PROGRAM] ---
# This step fine-tunes the prompts for your program.
qa_program = MedicalQASystem()

teleprompter = BootstrapFewShot(metric=None, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized_qa_program = teleprompter.compile(student=qa_program, trainset=trainset)


# --- [STEP 5: USE THE FINE-TUNED MODEL WITH A PDF] ---

# 1. Define the path to your PDF and your question
pdf_file_path = "abc (3).pdf"  # Make sure this file exists in your folder!
user_question = "What medication was prescribed to the patient?"

# 2. Parse the PDF to get its text content
document_content = parse_pdf_text(pdf_file_path)

if document_content:
    # 3. Run the optimized program with the extracted text
    prediction = optimized_qa_program(document_text=document_content, question=user_question)

    print(f"Analyzing Document: '{pdf_file_path}'")
    print(f"Question: {user_question}\n")
    print("--- Model Output ---")
    print(f"Predicted Medical Field: {prediction.medical_field}")
    print(f"Answer: {prediction.answer}")

    # Inspect the last prompt sent to the Gemini model
    #gemini_model.inspect_history(n=1)
else:
    print(f"Could not process the PDF file.")