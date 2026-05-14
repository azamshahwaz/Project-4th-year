# =========================================================
# PDF REPORT GENERATOR
# RESPONSIBLE AI PROJECT
# =========================================================

import os

from reportlab.platypus import (

    SimpleDocTemplate,

    Paragraph,

    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib.pagesizes import letter


# =========================================================
# GENERATE PDF REPORT
# =========================================================

def generate_pdf_report(
    output_path,
    content
):

    print(
        "\n========== PDF REPORT GENERATION =========="
    )

    try:

        # =================================================
        # CREATE OUTPUT DIRECTORY
        # =================================================

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # =================================================
        # CREATE PDF DOCUMENT
        # =================================================

        doc = SimpleDocTemplate(

            output_path,

            pagesize=letter
        )

        # =================================================
        # STYLES
        # =================================================

        styles = getSampleStyleSheet()

        elements = []

        # =================================================
        # TITLE
        # =================================================

        title = Paragraph(

            "<b>Responsible AI Report</b>",

            styles['Title']
        )

        elements.append(title)

        elements.append(
            Spacer(1, 20)
        )

        # =================================================
        # CONTENT
        # =================================================

        lines = content.split("\n")

        for line in lines:

            if line.strip() == "":

                elements.append(
                    Spacer(1, 10)
                )

                continue

            paragraph = Paragraph(

                line,

                styles['BodyText']
            )

            elements.append(paragraph)

        # =================================================
        # BUILD PDF
        # =================================================

        doc.build(elements)

        # =================================================
        # SUCCESS MESSAGE
        # =================================================

        print(
            f"\nPDF Report Saved:\n{output_path}"
        )

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "\nPDF Report Generation Failed"
        )

        print(e)