import subprocess

# Input and output file names
input_pdf = "../docs/szabo.pdf"
output_pdf = "szabo_ocr.pdf"

# Run OCR with ocrmypdf, showing live progress in the terminal
process = subprocess.Popen(
    [
        "ocrmypdf",
        "--skip-text",
        "--deskew",
        input_pdf,
        output_pdf
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Stream output line by line so you can see what's happening
for line in process.stdout:
    print(line, end="")

process.wait()

