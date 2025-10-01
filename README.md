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

1. Place your PDF file in the project folder (e.g., `abc (3).pdf`) in the Training data

2. Run the script:

```bash
streamlit run main.py
```

3. Output example:
"""
{
"Classification":"Principal Investigator Curriculum Vitae"
"DocumentDate":"04-29-2022"
"LabName":"ABC"
"ExpirationDate":"05-30-2025"
"ClassificationReason":"The document contains information about the education, professional experience, and clinical trial experience of a principal investigator."
"PersonnelName":"Carl Kuvalis"
"OrganizationName":"Syneos Health, New Chloe Memorial Medical Center, South Buckbury Memorial Hospital, San Jose Medical Center, Harvard Medical School, MIT"
"StudyName":"Alzheimer's, Parkinson's"
"Country":"USA"
"SiteNumber":"1007"
"Type":"Oncology"
"Subtype":"N/A"
}
"""
---

## Project Structure

```

├── main.py               # Main script to run PDF QA
├── requirements.txt      # Python dependencies 
├── training_data.py      # To train the model on files
├── file_extract.py       # To extract the file like the pdf,docx,....
├── README.md             # ReadME File
├── .env                  # Environment variables (API keys)

```

---

## How It Works

1. **Parse PDF:** `parse_pdf_text()` extracts all text from the PDF.
2. **Categorize:** `CategorizeMedicalDoc` predicts the medical field.
3. **Module Chaining:** `MedicalQASystem` chains categorization and QA.
4. **Few-Shot Prompt Optimization:** `BootstrapFewShot` improves accuracy using example data.

---

## Customization

* **Add More Medical Fields:** Update `medical_fields` list in `main.py`.
* **Add Training Examples:** Extend `train_data` for better few-shot performance.
* **Increase Token Limit:** Adjust `max_output_tokens` in the Gemini model for long PDFs.

---

## Requirements

* `Python 3.9+`
* `dspy`
* `PyMuPDF (fitz)`
* `Streamlit`
* `Google Gemini API access`
* `google-generativeai` 
* `python-dotenv `
* `pdfplumber `
* `pypdf2`
* `dspy `
* `PyMuPDF`
* `python-docx`
* `pandas`
* `openpyxl`
* `xlrd`
* `python-pptx`
* `odfpy`
* `pypandoc`
* `pytesseract`

