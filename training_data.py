import dspy
from file_extract import parse_pdf_text

train_data = [
    dspy.Example(
        document_text=parse_pdf_text("Trainset\SubI_CV_Ian_Miller_1007.pdf"),
    ).with_inputs("document_text"),
    dspy.Example(
        document_text=parse_pdf_text("Trainset\SubI_CV_Byron_Beer_1005.pdf"),
    ).with_inputs("document_text")
    
]