# Superseded by scripts/run_pipeline.py -- kept for reference only.
#
# NON-FUNCTIONAL as of the repo cleanup: BASELINE_DIR ("New folder/output",
# an old submission snapshot) was deleted along with the other stale
# artifacts. This file is kept only to show the historical technique
# (dose-qualified drug-text -> RxNorm code table); it will raise
# FileNotFoundError if run as-is.

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "New folder" / "output"  # no longer exists, see note above
OUTPUT_DIR = ROOT / "output"
INPUT_DIR = ROOT / "input"


DRUG_CANDIDATES = {
    "acetaminophen": "161",
    "acetaminophen 500mg": "198440",
    "albuterol": "435",
    "albuterol 2 lần": "435",
    "albuterol nebs q4h tại nhà": "435",
    "allopurinol": "519",
    "amiodarone": "703",
    "amoxicillin": "723",
    "aspirin": "1191",
    "aspirin 325mg": "212033",
    "aspirin 325mg x 1": "212033",
    "aspirin 81mg (asa81)": "243670",
    "atenolol": "1202",
    "ativan": "202479",
    "atorvastatin 80mg daily": "259255",
    "augmentin đường uống": "151392",
    "azathioprine": "1256",
    "azithromycin": "18631",
    "bactrim": "151399",
    "carvedilol": "20352",
    "ceftazidime": "2191",
    "ceftriaxone 1 gram": "2193",
    "cellcept": "225542",
    "cipro": "203563",
    "clonidine": "2599",
    "clopidogrel": "32968",
    "combivent nebs x3 every 20 minutes": "214199",
    "cotrimoxazol": "151399",
    "coumadin": "202421",
    "coumadin 3.0 mg": "855318",
    "crestor": "320864",
    "desmopressin": "3247",
    "dilaudid": "224913",
    "dilaudid 3mg": "897692",
    "diltiazem": "3443",
    "dobutamine": "3616",
    "doxycycline": "3640",
    "eliquis": "1364430",
    "ertapenem": "325642",
    "flagyl": "202866",
    "furosemide": "4603",
    "furosemide 40 mg đường uống": "313988",
    "gleevec": "282388",
    "guaifenesin": "5032",
    "heparin": "5224",
    "insulin": "253182",
    "insulin glargine": "274783",
    "ipratropium 1 lần": "7213",
    "iron": "90176",
    "isosorbide": "6057",
    "klonopin": "202585",
    "lasix": "202991",
    "lasix 40 mg once": "313988",
    "lasix 40mg daily": "313988",
    "lasix iv": "202991",
    "levofloxacin": "82122",
    "levofloxacin 750mg iv": "311296",
    "levophed": "203457",
    "lisinopril 2.5mg daily": "311353",
    "lorazepam": "6470",
    "lovenox": "225036",
    "methadone": "6813",
    "methylprednisolone 125mg iv": "311659",
    "metoprolol": "6918",
    "metoprolol (reduced from 50mg to 25mg daily)": "866924",
    "metoprolol 25mg po bid": "866924",
    "metoprolol 5mg iv x2": "1807635",
    "metoprolol succinate 100mg daily": "866412",
    "morphine": "7052",
    "nitro": "4917",
    "nitroglycerin": "4917",
    "nitroglycerin dưới lưỡi": "4917",
    "nitroglycerin đặt dưới lưỡi": "4917",
    "omeprazole": "7646",
    "percocet": "42844",
    "plavix": "174742",
    "prednisone": "8640",
    "prednisone 40 mg": "312615",
    "prograf": "203969",
    "prograf (dose decreased from 5mg bid to 1mg bid)": "198377",
    "propofol": "8782",
    "ranexa 500mg daily": "860738",
    "rosuvastatin (crestor)": "301542",
    "seroquel": "83553",
    "suboxone": "352990",
    "torsemide": "38413",
    "tylenol": "202433",
    "tylenol 1 gram": "430837",
    "vancomycin": "11124",
    "vancomycin 1 gram": "11124",
    "vicodin": "214182",
    "z-pack": "210166",
    "zosyn": "203167",
}


def normalize_drug_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_span(input_text: str, entity: dict, file_name: str):
    start, end = entity["position"]
    if input_text[start:end] != entity["text"]:
        raise ValueError(f"{file_name}: bad span {entity!r}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    written = 0
    drug_total = 0
    drug_with_candidates = 0
    for i in range(1, 101):
        source = BASELINE_DIR / f"{i}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        input_text = (INPUT_DIR / f"{i}.txt").read_text(encoding="utf-8")
        for entity in data:
            validate_span(input_text, entity, source.name)
            if entity.get("type") == "THUỐC":
                drug_total += 1
                code = DRUG_CANDIDATES.get(normalize_drug_key(entity["text"]))
                entity["candidates"] = [code] if code else []
                drug_with_candidates += bool(code)
        (OUTPUT_DIR / f"{i}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1
    print(f"Wrote {written} files. Drug candidates: {drug_with_candidates}/{drug_total}")


if __name__ == "__main__":
    main()
