from pathlib import Path
import runpy


if __name__ == "__main__":
    phase_one_script = Path(__file__).parent / "Faza I" / "phase1_pipeline.py"
    runpy.run_path(str(phase_one_script), run_name="__main__")
