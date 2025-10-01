import dspy
from file_extract import parse_pdf_text
import os

# List of your PDF paths
pdf_paths = [
    r"Trainset/1007/PI_CV_Peter_Labadie_1007.pdf",
    r"Trainset/1007\SubI_CV_Ian_Miller_1007.pdf",
    r"Trainset/1007/Training_Log_04-Jun-2025_Site_1007.pdf",
    r"Trainset/1007/Training_Log_05-Jun-2025_Site_1007.pdf",
    r"Trainset/1007/Training_Protocol_Training_Summary_1007.pdf"
]

# Build train_data with a loop
train_data = []
for path in pdf_paths:
    text = parse_pdf_text(path)
    example = dspy.Example(document_text=text).with_inputs("document_text")
    train_data.append(example)



#---Other way to write the above code--#

# import dspy
# from file_extract import parse_pdf_text

# train_data = [
#     dspy.Example(
#         document_text=parse_pdf_text("Trainset\SubI_CV_Ian_Miller_1007.pdf"),
#     ).with_inputs("document_text"),
#     dspy.Example(
#         document_text=parse_pdf_text("Trainset\SubI_CV_Byron_Beer_1005.pdf"),
#     ).with_inputs("document_text")
    
# ]
