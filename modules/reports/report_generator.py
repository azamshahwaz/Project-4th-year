import os

def generate_report():

    os.makedirs(

        "outputs/reports",

        exist_ok=True
    )

    report_path = (
        "outputs/reports/final_report.txt"
    )

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "RESPONSIBLE AI REPORT\n"
        )

        f.write(
            "\nPipeline Completed Successfully"
        )

    print(
        "\nFinal Report Generated"
    )