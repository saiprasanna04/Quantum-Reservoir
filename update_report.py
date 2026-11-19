"""Update the Word report's result table and insert generated plots after running the experiment."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "results.json"
REPORT = ROOT / "Quantum_Reservoir_Computing_Project_Report.docx"

def main():
    if not RESULTS.exists():
        raise SystemExit("Run `python src/run_experiment.py` first; results/results.json was not found.")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    doc = Document(REPORT)

    # Update the first table (model metrics).
    table = doc.tables[0]
    row_map = {
        "Persistence": data["metrics"]["Persistence"],
        "Classical Ridge": data["metrics"]["Classical Ridge"],
        "Quantum Reservoir": data["metrics"]["Quantum Reservoir"],
    }
    for row in table.rows[1:]:
        name = row.cells[0].text.strip()
        if name in row_map:
            m = row_map[name]
            row.cells[1].text = f'{m["MAE"]:.6f}'
            row.cells[2].text = f'{m["RMSE"]:.6f}'
            row.cells[3].text = f'{m["R2"]:.6f}'

    # Add a concise measured-results paragraph after the result table.
    p = doc.add_paragraph()
    p.add_run(
        f'Run-generated configuration: {data["n_points"]} samples; '
        f'{data["n_qubits"]} qubits; window size {data["window_size"]}; '
        f'chronological split {data["train_validation_test_split"]}; '
        f'selected Ridge alpha {data["best_ridge_alpha"]}; '
        f'total runtime {data["runtime_seconds"]:.3f} seconds. '
        'These results come from an ideal simulator and are not physical trapped-ion hardware results.'
    )

    for filename, caption in [
        ("forecast_comparison.png", "Figure 1. Actual and predicted test-set values."),
        ("rmse_comparison.png", "Figure 2. RMSE comparison across the evaluated models."),
    ]:
        image_path = ROOT / "figures" / filename
        if image_path.exists():
            cp = doc.add_paragraph(caption)
            cp.alignment = 1
            ip = doc.add_paragraph()
            ip.alignment = 1
            ip.add_run().add_picture(str(image_path), width=Inches(6.0))

    doc.save(REPORT)
    print(f"Updated report: {REPORT}")

if __name__ == "__main__":
    main()
