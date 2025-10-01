This project provides a **Medical PDF Question-Answering (QA) system** that:

- Extracts text from medical PDF documents
- Categorizes the document into a medical field (e.g., Cardiology, Oncology)
- Answers questions based strictly on the document content
- Uses **Google Gemini** via **DSPy** for few-shot reasoning and QA

---

## Features

- **PDF Parsing:** Extract text from PDFs using `PyMuPDF (fitz)`
- **Medical Field Classification:** Categorizes PDFs into predefined medical specialties
- **Contextual QA:** Answers questions using only the PDF content
- **Few-Shot Fine-Tuning:** Optimizes prompts using `DSPy`'s `BootstrapFewShot`
- **Modular Design:** Clean separation of categorization, QA, and module logic

---

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (if not already created) and add your Google Gemini API key:

```dotenv
GEMINI_API_KEY="your_api_key_here"
```

> The script automatically loads the API key from `.env`.

---

## Usage

1. Place your PDF file in the project folder (e.g., `abc (3).pdf`).
2. Update the `pdf_file_path` and `user_question` variables in your script:

```python
pdf_file_path = "abc (3).pdf"
user_question = "What medication was prescribed to the patient?"
```

3. Run the script:

```bash
streamlit run main.py
```

4. Output example:

```
Analyzing Document: 'abc (3).pdf'
Question: What medication was prescribed to the patient?

--- Model Output ---
Predicted Medical Field: Cardiology
Answer: Beta-blockers and anticoagulants.
```

---

## Project Structure

```
.
├── main.py               # Main script to run PDF QA
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── README.md             # This documentation
└── abc (3).pdf           # Sample PDF (replace with your own)
```

---

## How It Works

1. **Parse PDF:** `parse_pdf_text()` extracts all text from the PDF.
2. **Categorize:** `CategorizeMedicalDoc` predicts the medical field.
3. **Answer Questions:** `AnswerFromDocument` answers questions based on PDF content.
4. **Module Chaining:** `MedicalQASystem` chains categorization and QA.
5. **Few-Shot Prompt Optimization:** `BootstrapFewShot` improves accuracy using example data.

---

## Customization

* **Add More Medical Fields:** Update `medical_fields` list in `main.py`.
* **Add Training Examples:** Extend `train_data` for better few-shot performance.
* **Increase Token Limit:** Adjust `max_output_tokens` in the Gemini model for long PDFs.

---

## Requirements

* Python 3.9+
* `dspy`
* `PyMuPDF (fitz)`
* Google Gemini API access

Do you want me to add that?
```
