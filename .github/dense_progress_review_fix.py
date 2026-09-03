from pathlib import Path

root = Path(__file__).resolve().parents[1]

def replace_once(path: str, old: str, new: str) -> None:
    target = root / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match in {path}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    "web/src/lib/domain/garagePreview.ts",
    """function progressNumber(record: Record<string, unknown>, key: string): number | null {\n  const value = Number(record[key]);\n  return Number.isFinite(value) ? value : null;\n}\n""",
    """function progressNumber(record: Record<string, unknown>, key: string): number | null {\n  const raw = record[key];\n  if (raw === null || raw === undefined || raw === '') return null;\n  const value = Number(raw);\n  return Number.isFinite(value) ? value : null;\n}\n""",
)
replace_once(
    "web/tests/garage-preview.test.mjs",
    """    actual: { traffic: 64, walkers: 24, pedestrian_crossing_factor: 0.45 },\n""",
    """    actual: { traffic: 64, walkers: 24, pedestrian_crossing_factor: null },\n""",
)
