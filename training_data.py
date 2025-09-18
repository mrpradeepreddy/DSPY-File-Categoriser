import dspy
from file_extract import parse_pdf_text

train_data = [
    dspy.Example(
        document_text=parse_pdf_text("Trainset\cardio_train\cardio_train_1.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),
    dspy.Example(
        document_text=parse_pdf_text("Trainset\cardio_train\cardio_train_2.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\cardio_train\cardio_train_3.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\cardio_train\cardio_train_4.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

# Dermatology
    dspy.Example(
        document_text=parse_pdf_text("Trainset\Dermatology\derma_train_1.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Dermatology\derma_train_2.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Dermatology\derma_train_3.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Dermatology\derma_train_4.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

# Neurology

    dspy.Example(
        document_text=parse_pdf_text("Trainset/Neurology/neuro_train_1.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset/Neurology/neuro_train_2.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset/Neurology/neuro_train_3.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset/Neurology/neuro_train_4.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

# Oncology

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Oncology\onco_train_1.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Oncology\onco_train_2.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Oncology\onco_train_3.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Oncology\onco_train_4.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

# Pediatrics

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Pediatrics\peds_train_1.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Pediatrics\peds_train_2.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Pediatrics\peds_train_3.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    dspy.Example(
        document_text=parse_pdf_text("Trainset\Pediatrics\peds_train_4.pdf"),
        medical_field="Cardiology",
    ).with_inputs("document_text"),

    
]