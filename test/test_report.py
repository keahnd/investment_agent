from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=16)
pdf.cell(0, 10, "Test Report", ln=True)
pdf.set_font("Helvetica", size=12)
pdf.cell(0, 10, "If this renders, fpdf2 is working.", ln=True)
pdf.output("test_output.pdf")
print("PDF generated successfully")