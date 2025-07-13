import PyPDF2


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        # Open the PDF file in binary mode
        with open(pdf_path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Print number of pages
            print(f"Number of pages: {len(pdf_reader.pages)}")

            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n\n--- Page {page_num + 1} ---\n\n"
                text += page.extract_text()

        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


# Path to the PDF file
pdf_path = "/Users/karlkompatscher/Dev/InventoryManagement/Folien/IM_Formula_Sheet.pdf"

# Extract text from the PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Print the extracted text
print(extracted_text)

# Also try the other PDF
pdf_path2 = (
    "/Users/karlkompatscher/Dev/InventoryManagement/Exam/studocu/formula-sheet-im.pdf"
)
print("\n\n=== SECOND PDF ===\n\n")
extracted_text2 = extract_text_from_pdf(pdf_path2)
print(extracted_text2)
