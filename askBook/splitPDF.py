import os
from pikepdf import Pdf


def splitChapters():
    file_path = "pdfs/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
    chaptersDict = {
        0: [0, 19],  # pages 1 to 20 - Front Matter
        1: [20, 85],  # pages 21 to 86 - Chapter 1
        2: [86, 155],  # pages 87 to 156 - Chapter 2
        # ... and so on add more chapters
    }

    pdf = Pdf.open(file_path)
    new_pdf_files = [Pdf.new() for i in chaptersDict]
    new_pdf_index = 0

    for new_pdf_index, (startPage, endPage) in chaptersDict.items():
        for n in range(startPage, endPage + 1):
            new_pdf_files[new_pdf_index].pages.append(pdf.pages[n])

        pdfName, extension = os.path.splitext(file_path)
        output_filename = f"{pdfName}-{new_pdf_index}.pdf"
        new_pdf_files[new_pdf_index].save(output_filename)
        print(f"[+] File: {output_filename} saved.")


splitChapters()
