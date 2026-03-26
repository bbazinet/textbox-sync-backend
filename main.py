from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BusinessTextBox PMS Sync API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROPERTY_CONFIGS: Dict[str, dict] = {
    "default_letter": {
        "contact2_template": "apt {unit}",
        "groups_template": "#all#bldg{building}",
        "building_strategy": "first_char",
    },
    "default_numeric": {
        "contact2_template": "apt {unit}",
        "groups_template": "#all#bldg{building}",
        "building_strategy": "before_dash",
    },
}


class PreviewResponse(BaseModel):
    job_id: str
    totals: dict
    invalid_rows: List[dict]
    adds_preview: List[dict]
    updates_preview: List[dict]
    removals_preview: List[dict]
    downloads: dict


@dataclass
class SyncArtifacts:
    import_path: Path
    removals_path: Optional[Path]
    summary_path: Path


def load_table_from_upload(upload: UploadFile) -> pd.DataFrame:
    suffix = Path(upload.filename or "upload").suffix.lower()
    raw = upload.file.read()
    buffer = io.BytesIO(raw)

    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        return pd.read_excel(buffer, engine=engine)

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def detect_and_clean_onesite(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    header_row = None

    for i in range(min(len(working), 25)):
        row_values = [str(v).strip().lower() for v in working.iloc[i].tolist()]
        if "name" in row_values and any("bldg/unit" == v for v in row_values):
            header_row = i
            break

    if header_row is None:
        raise HTTPException(status_code=400, detail="Could not detect OneSite header row.")

    working = working.iloc[header_row:].copy()
    working.columns = working.iloc[0]
    working = working.iloc[1:].reset_index(drop=True)
    return working


def clean_phone(value: object) -> str:
    digits = re.sub(r"\D", "", "" if pd.isna(value) else str(value))
    if len(digits) != 10:
        return ""
    if digits == "9999999999":
        return ""
    return digits


def title_case_name(name: object) -> str:
    value = "" if pd.isna(name) else str(name).strip()
    return value.title()


def extract_building(unit_value: str, strategy: str) -> str:
    unit_value = (unit_value or "").strip()
    if not unit_value:
        return ""

    if strategy == "first_char":
        return unit_value[:1]
    if strategy == "first_two" and len(unit_value) >= 2:
        return unit_value[:2]
    if strategy == "before_dash":
        return unit_value.split("-")[0]

    return unit_value[:1]


def extract_floor(unit_value: str, strategy: str) -> str:
    unit_value = (unit_value or "").strip()
    if not unit_value:
        return ""

    if strategy == "third_char" and len(unit_value) >= 3:
        return unit_value[2]

    return ""


def split_name_parts(name: str) -> Tuple[str, str, str]:
    value = name.strip()
    if not value:
        return "", "", ""

    if "," in value:
        last, rest = [p.strip() for p in value.split(",", 1)]
        tokens = [t for t in rest.split() if t]
        first = tokens[0] if tokens else ""
        remaining = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        return first, remaining, last

    tokens = [t for t in value.split() if t]
    if len(tokens) == 1:
        return tokens[0], "", ""

    first = tokens[0]
    last = tokens[-1]
    remaining = " ".join(tokens[1:-1]) if len(tokens) > 2 else ""
    return first, remaining, last


def merge_household_names(names: List[str]) -> str:
    ordered_unique = []
    seen = set()
    for name in names:
        normalized = re.sub(r"\s+", " ", name.strip())
        if normalized and normalized.lower() not in seen:
            ordered_unique.append(normalized)
            seen.add(normalized.lower())

    if not ordered_unique:
        return ""
    if len(ordered_unique) == 1:
        return ordered_unique[0]

    parsed = [split_name_parts(name) for name in ordered_unique]
    last_names = [last for _, _, last in parsed if last]

    if len(last_names) == len(parsed) and len({ln.lower() for ln in last_names}) == 1:
        shared_last = last_names[0]
        has_comma_format = all("," in name for name in ordered_unique)

        if has_comma_format:
            first_segments = []
            for first, remaining, _ in parsed:
                segment = " ".join([first, remaining]).strip()
                first_segments.append(segment)
            return f"{shared_last}, {' & '.join(first_segments)}"

        first_segments = []
        for first, remaining, _ in parsed:
            segment = " ".join([first, remaining]).strip()
            first_segments.append(segment)
        return f"{' & '.join(first_segments)} {shared_last}".strip()

    return " & ".join(ordered_unique)


def normalize_pms_export(raw_df: pd.DataFrame, property_config: dict) -> Tuple[pd.DataFrame, List[dict]]:
    cleaned = detect_and_clean_onesite(raw_df)

    required_columns = ["Name", "Bldg/Unit", "Phone"]
    missing = [col for col in required_columns if col not in cleaned.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required PMS columns: {missing}")

    working = cleaned[["Name", "Bldg/Unit", "Phone"]].copy()
    working["source_row"] = range(1, len(working) + 1)
    working["Contact1"] = working["Name"].apply(title_case_name)
    working["Phone"] = working["Phone"].apply(clean_phone)

    working["Unit"] = working["Bldg/Unit"].fillna("").astype(str).str.strip()
    working["Building"] = working["Unit"].apply(lambda x: extract_building(x, property_config["building_strategy"]))
    working["Floor"] = working["Unit"].apply(lambda x: extract_floor(x, property_config.get("floor_strategy", "none")))

    working["Contact2"] = working.apply(
        lambda row: property_config["contact2_template"].format(
            unit=row["Unit"],
            building=row["Building"],
            floor=row["Floor"],
        ),
        axis=1,
    )
    working["Groups"] = working.apply(
        lambda row: property_config["groups_template"].format(
            unit=row["Unit"],
            building=row["Building"],
            floor=row["Floor"],
        ),
        axis=1,
    )

    invalid_mask = (
        (working["Phone"] == "")
        | (working["Contact1"].str.strip() == "")
        | (working["Contact2"].str.strip() == "")
        | (working["Groups"].str.strip() == "")
    )

    invalid_rows = []
    for _, row in working[invalid_mask].iterrows():
        reason = []
        if not row["Phone"]:
            reason.append("Invalid phone")
        if not str(row["Contact1"]).strip():
            reason.append("Blank Contact1")
        if not str(row["Contact2"]).strip():
            reason.append("Blank Contact2")
        if not str(row["Groups"]).strip():
            reason.append("Blank Groups")

        invalid_rows.append(
    {
        "row": int(row["source_row"]),
        "name": "" if pd.isna(row["Name"]) else str(row["Name"]),
        "phone": re.sub(r"\D", "", "" if pd.isna(row["Phone"]) else str(row["Phone"])),
        "reason": ", ".join(reason),
    }
)

    valid = working[~invalid_mask].copy()

    merged_records = []
    for (phone, unit), group in valid.groupby(["Phone", "Unit"], dropna=False, sort=False):
        merged_name = merge_household_names(group["Contact1"].tolist())
        merged_records.append(
            {
                "Contact1": merged_name,
                "Contact2": group["Contact2"].iloc[0],
                "Phone": phone,
                "Groups": group["Groups"].iloc[0],
                "Unit": unit,
                "Building": group["Building"].iloc[0],
                "resident_count": int(group["Contact1"].nunique()),
            }
        )

    output = pd.DataFrame(merged_records)
    output = output[(output["Contact1"] != "") & (output["Contact2"] != "") & (output["Groups"] != "")]
    output = output[["Contact1", "Contact2", "Phone", "Groups", "Unit", "Building", "resident_count"]]
    return output, invalid_rows


def normalize_current_contacts(current_df: pd.DataFrame) -> pd.DataFrame:
    column_map = {c.lower(): c for c in current_df.columns}
    if "phonenumber" not in column_map:
        raise HTTPException(status_code=400, detail="Current TextBox export must include PhoneNumber.")

    phone_col = column_map["phonenumber"]
    contact1_col = column_map.get("contact1")
    contact2_col = column_map.get("contact2")
    groups_col = column_map.get("groups")

    normalized = pd.DataFrame()
    normalized["Phone"] = current_df[phone_col].astype(str).str.replace(r"\D", "", regex=True)
    normalized["Contact1"] = current_df[contact1_col].fillna("").astype(str) if contact1_col else ""
    normalized["Contact2"] = current_df[contact2_col].fillna("").astype(str) if contact2_col else ""
    normalized["Groups"] = current_df[groups_col].fillna("").astype(str) if groups_col else ""
    normalized = normalized[normalized["Phone"].str.match(r"^\d{10}$", na=False)].copy()
    normalized = normalized.drop_duplicates(subset=["Phone"], keep="first")
    return normalized

def infer_apartment_format_from_textbox(current_df: pd.DataFrame) -> dict:
    normalized = normalize_current_contacts(current_df)

    sample = normalized[
        (normalized["Contact2"].fillna("").astype(str).str.strip() != "")
        & (normalized["Groups"].fillna("").astype(str).str.strip() != "")
    ].copy()

    if sample.empty:
        return {
            "contact2_template": "apt {unit}",
            "groups_template": "#all#bldg{building}",
            "building_strategy": "first_char",
            "floor_strategy": "none",
        }

    sample["Contact2"] = sample["Contact2"].astype(str).str.strip()
    sample["Groups"] = sample["Groups"].astype(str).str.strip()

    # Detect if Contact2 usually looks like "apt XXXXX"
    apt_prefixed = sample["Contact2"].str.lower().str.startswith("apt ").mean() >= 0.7

    # Try to extract unit from Contact2
    if apt_prefixed:
        units = sample["Contact2"].str[4:].str.strip()
        contact2_template = "apt {unit}"
    else:
        units = sample["Contact2"].str.strip()
        contact2_template = "{unit}"

    # Detect building/floor style from groups and unit
    floor_pattern_ratio = sample["Groups"].str.contains(r"#bldg\d{2}floor\d", case=False, regex=True).mean()
    two_digit_bldg_ratio = sample["Groups"].str.contains(r"#bldg\d{2}", case=False, regex=True).mean()

    # Default assumptions
    building_strategy = "first_char"
    floor_strategy = "none"
    groups_template = "#all#bldg{building}"

    # Compact numeric unit pattern like 01101, 08206, 14303
    compact_numeric_ratio = units.str.match(r"^\d{5}$", na=False).mean()

    if compact_numeric_ratio >= 0.7 and two_digit_bldg_ratio >= 0.7:
        building_strategy = "first_two"
        groups_template = "#all#bldg{building}"

        if floor_pattern_ratio >= 0.5:
            floor_strategy = "third_char"
            groups_template = "#all#bldg{building}#bldg{building}floor{floor}"

    # Dashed apartment pattern like 207-101
    dashed_ratio = units.str.contains(r"^\d+-\d+$", regex=True, na=False).mean()
    if dashed_ratio >= 0.7:
        building_strategy = "before_dash"
        floor_strategy = "none"
        groups_template = "#all#bldg{building}"

    # Letter building pattern like A206
    letter_ratio = units.str.contains(r"^[A-Za-z]\d+", regex=True, na=False).mean()
    if letter_ratio >= 0.7:
        building_strategy = "first_char"
        floor_strategy = "none"
        groups_template = "#all#bldg{building}"

    return {
        "contact2_template": contact2_template,
        "groups_template": groups_template,
        "building_strategy": building_strategy,
        "floor_strategy": floor_strategy,
    }

def diff_contacts(new_df: pd.DataFrame, old_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if old_df is None or old_df.empty:
        adds = new_df.copy()
        updates = pd.DataFrame(columns=["Phone", "old_contact1", "new_contact1", "old_contact2", "new_contact2", "old_groups", "new_groups", "changed_fields"])
        unchanged = pd.DataFrame(columns=new_df.columns)
        removals = pd.DataFrame(columns=["Phone", "Contact1", "Contact2", "Groups"])
        return adds, updates, unchanged, removals

    old_lookup = old_df.set_index("Phone", drop=False)
    new_lookup = new_df.set_index("Phone", drop=False)

    add_rows = []
    update_rows = []
    unchanged_rows = []

    for phone, row in new_lookup.iterrows():
        if phone not in old_lookup.index:
            add_rows.append(row.to_dict())
            continue

        old_row = old_lookup.loc[phone]
        changed_fields = []
        if str(old_row.get("Contact1", "")).strip() != str(row.get("Contact1", "")).strip():
            changed_fields.append("Contact1")
        if str(old_row.get("Contact2", "")).strip() != str(row.get("Contact2", "")).strip():
            changed_fields.append("Contact2")
        if str(old_row.get("Groups", "")).strip() != str(row.get("Groups", "")).strip():
            changed_fields.append("Groups")

        if changed_fields:
            update_rows.append(
                {
                    "Phone": phone,
                    "old_contact1": old_row.get("Contact1", ""),
                    "new_contact1": row.get("Contact1", ""),
                    "old_contact2": old_row.get("Contact2", ""),
                    "new_contact2": row.get("Contact2", ""),
                    "old_groups": old_row.get("Groups", ""),
                    "new_groups": row.get("Groups", ""),
                    "changed_fields": ", ".join(changed_fields),
                }
            )
        else:
            unchanged_rows.append(row.to_dict())

    removals = old_df[~old_df["Phone"].isin(new_df["Phone"])].copy()

    adds = pd.DataFrame(add_rows)
    updates = pd.DataFrame(update_rows)
    unchanged = pd.DataFrame(unchanged_rows)
    return adds, updates, unchanged, removals


def build_full_sync_file(new_df: pd.DataFrame, old_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        output = new_df.copy()
        output = output.rename(columns={"Phone": "PhoneNumber"})
        return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]

    old_lookup = old_df.set_index("Phone", drop=False)
    new_lookup = new_df.set_index("Phone", drop=False)

    final_rows = []

    # Current residents from the new PMS file
    for phone, row in new_lookup.iterrows():
        final_rows.append(
            {
                "Contact1": row.get("Contact1", ""),
                "Contact2": row.get("Contact2", ""),
                "Phone": phone,
                "Groups": row.get("Groups", ""),
            }
        )

    # Past residents from the old TextBox file
    for phone, row in old_lookup.iterrows():
        if phone not in new_lookup.index:
            final_rows.append(
                {
                    "Contact1": row.get("Contact1", ""),
                    "Contact2": "",
                    "Phone": phone,
                    "Groups": "",
                }
            )

    output = pd.DataFrame(final_rows)
    output = output.rename(columns={"Phone": "PhoneNumber"})
    return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]

def build_delta_sync_file(new_df: pd.DataFrame, old_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        output = new_df.copy()
        output = output.rename(columns={"Phone": "PhoneNumber"})
        return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]

    old_lookup = old_df.set_index("Phone", drop=False)
    new_lookup = new_df.set_index("Phone", drop=False)

    final_rows = []

    # Adds + updates
    for phone, row in new_lookup.iterrows():
        if phone not in old_lookup.index:
            final_rows.append(
                {
                    "Contact1": row.get("Contact1", ""),
                    "Contact2": row.get("Contact2", ""),
                    "Phone": phone,
                    "Groups": row.get("Groups", ""),
                }
            )
            continue

        old_row = old_lookup.loc[phone]

        old_contact1 = str(old_row.get("Contact1", "")).strip()
        old_contact2 = str(old_row.get("Contact2", "")).strip()
        old_groups = str(old_row.get("Groups", "")).strip()

        new_contact1 = str(row.get("Contact1", "")).strip()
        new_contact2 = str(row.get("Contact2", "")).strip()
        new_groups = str(row.get("Groups", "")).strip()

        if (
            old_contact1 != new_contact1
            or old_contact2 != new_contact2
            or old_groups != new_groups
        ):
            final_rows.append(
                {
                    "Contact1": new_contact1,
                    "Contact2": new_contact2,
                    "Phone": phone,
                    "Groups": new_groups,
                }
            )

    # Deactivations: only if something actually needs to be cleared
    for phone, row in old_lookup.iterrows():
        if phone not in new_lookup.index:
            old_contact2 = str(row.get("Contact2", "")).strip()
            old_groups = str(row.get("Groups", "")).strip()

            if old_contact2 or old_groups:
                final_rows.append(
                    {
                        "Contact1": str(row.get("Contact1", "")).strip(),
                        "Contact2": "",
                        "Phone": phone,
                        "Groups": "",
                    }
                )

    output = pd.DataFrame(final_rows)

    if output.empty:
        output = pd.DataFrame(columns=["Contact1", "Contact2", "PhoneNumber", "Groups"])
        return output

    output = output.drop_duplicates(subset=["Phone"], keep="first")
    output = output.rename(columns={"Phone": "PhoneNumber"})
    return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]

def build_general_full_sync_file(new_df: pd.DataFrame, old_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        output = new_df.copy()
        output = output.rename(columns={"Phone": "PhoneNumber"})
        return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]

    old_lookup = old_df.set_index("Phone", drop=False)
    new_lookup = new_df.set_index("Phone", drop=False)

    final_rows = []

    # New + updated contacts
    for phone, row in new_lookup.iterrows():
        final_rows.append(
            {
                "Contact1": row.get("Contact1", ""),
                "Contact2": row.get("Contact2", ""),
                "Phone": phone,
                "Groups": row.get("Groups", ""),
            }
        )

    # Existing contacts NOT in new file → KEEP AS-IS
    for phone, row in old_lookup.iterrows():
        if phone not in new_lookup.index:
            final_rows.append(
                {
                    "Contact1": row.get("Contact1", ""),
                    "Contact2": row.get("Contact2", ""),
                    "Phone": phone,
                    "Groups": row.get("Groups", ""),
                }
            )

    output = pd.DataFrame(final_rows)
    output = output.drop_duplicates(subset=["Phone"], keep="first")
    output = output.rename(columns={"Phone": "PhoneNumber"})
    return output[["Contact1", "Contact2", "PhoneNumber", "Groups"]]


def build_summary(
    cleaned_df: pd.DataFrame,
    invalid_rows: List[dict],
    adds: pd.DataFrame,
    updates: pd.DataFrame,
    unchanged: pd.DataFrame,
    removals: pd.DataFrame,
) -> dict:
    return {
        "uploadedRows": int(len(cleaned_df) + len(invalid_rows)),
        "validContacts": int(len(cleaned_df)),
        "invalidPhones": int(sum("Invalid phone" in row["reason"] for row in invalid_rows)),
        "duplicateMerges": int(sum(max(0, int(x) - 1) for x in cleaned_df.get("resident_count", pd.Series(dtype=int)).fillna(1))),
        "skippedRows": int(len(invalid_rows)),
        "adds": int(len(adds)),
        "updates": int(len(updates)),
        "unchanged": int(len(unchanged)),
        "removals": int(len(removals)),
    }


def write_artifacts(job_id: str, cleaned_df: pd.DataFrame, removals: pd.DataFrame, summary: dict) -> SyncArtifacts:
    job_dir = ARTIFACT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    import_path = job_dir / "textbox_import.xlsx"
    summary_path = job_dir / "sync_summary.xlsx"
    removals_path = job_dir / "contacts_to_remove.xlsx"

    cleaned_df[["Contact1", "Contact2", "Phone", "Groups"]].to_excel(import_path, index=False)

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

    if removals is not None and not removals.empty:
        removals[["Phone", "Contact1", "Contact2", "Groups"]].to_excel(removals_path, index=False)
    else:
        pd.DataFrame(columns=["Phone", "Contact1", "Contact2", "Groups"]).to_excel(removals_path, index=False)

    return SyncArtifacts(import_path=import_path, removals_path=removals_path, summary_path=summary_path)


@app.get("/health")
def health_check():
    return {"ok": True}


@app.post("/sync/preview", response_model=PreviewResponse)
async def sync_preview(
    property_key: str = Form("default_letter"),
    pms_file: UploadFile = File(...),
    current_contacts_file: Optional[UploadFile] = File(None),
):
    
    if property_key not in PROPERTY_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown property config: {property_key}")

    raw_df = load_table_from_upload(pms_file)
    current_df = load_table_from_upload(current_contacts_file) if current_contacts_file else None

    inferred_config = PROPERTY_CONFIGS[property_key].copy()

    if current_df is not None:
        inferred_config.update(infer_apartment_format_from_textbox(current_df))

    cleaned_df, invalid_rows = normalize_pms_export(raw_df, inferred_config)
    old_df = normalize_current_contacts(current_df) if current_df is not None else None

    delta_sync_df = build_delta_sync_file(cleaned_df, old_df)
    adds, updates, unchanged, removals = diff_contacts(cleaned_df, old_df)

    summary = build_summary(cleaned_df, invalid_rows, adds, updates, unchanged, removals)
    job_id = uuid.uuid4().hex

    job_dir = ARTIFACT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    import_path = job_dir / "textbox_delta_sync.xlsx"
    summary_path = job_dir / "sync_summary.xlsx"

    delta_sync_df.to_excel(import_path, index=False)

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

    payload = {
        "job_id": job_id,
        "totals": summary,
        "invalid_rows": invalid_rows[:100],
        "adds_preview": adds[[c for c in ["Contact1", "Contact2", "Phone", "Groups"] if c in adds.columns]].head(25).fillna("").to_dict(orient="records"),
        "updates_preview": updates.head(25).fillna("").to_dict(orient="records"),
        "removals_preview": removals[[c for c in ["Contact1", "Contact2", "Phone", "Groups"] if c in removals.columns]].head(25).fillna("").to_dict(orient="records"),
        "downloads": {
            "import": f"/sync/download/{job_id}/textbox_delta_sync.xlsx",
            "summary": f"/sync/download/{job_id}/sync_summary.xlsx",
        },
    }
    return JSONResponse(payload)


@app.post("/sync/general-preview", response_model=PreviewResponse)
async def general_sync_preview(
    property_key: str = Form("default_letter"),
    pms_file: UploadFile = File(...),
    current_contacts_file: Optional[UploadFile] = File(None),
):
    if property_key not in PROPERTY_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown property config: {property_key}")

    raw_df = load_table_from_upload(pms_file)
    current_df = load_table_from_upload(current_contacts_file) if current_contacts_file else None

    cleaned_df, invalid_rows = normalize_pms_export(raw_df, PROPERTY_CONFIGS[property_key])
    old_df = normalize_current_contacts(current_df) if current_df is not None else None

    full_sync_df = build_general_full_sync_file(cleaned_df, old_df)
    adds, updates, unchanged, removals = diff_contacts(cleaned_df, old_df)

    summary = build_summary(cleaned_df, invalid_rows, adds, updates, unchanged, removals)
    job_id = uuid.uuid4().hex

    job_dir = ARTIFACT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    import_path = job_dir / "textbox_general_full_sync.xlsx"
    summary_path = job_dir / "sync_summary.xlsx"

    full_sync_df.to_excel(import_path, index=False)

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

    payload = {
        "job_id": job_id,
        "totals": summary,
        "invalid_rows": invalid_rows[:100],
        "adds_preview": adds[[c for c in ["Contact1", "Contact2", "Phone", "Groups"] if c in adds.columns]].head(25).fillna("").to_dict(orient="records"),
        "updates_preview": updates.head(25).fillna("").to_dict(orient="records"),
        "removals_preview": removals[[c for c in ["Contact1", "Contact2", "Phone", "Groups"] if c in removals.columns]].head(25).fillna("").to_dict(orient="records"),
        "downloads": {
            "import": f"/sync/download/{job_id}/textbox_general_full_sync.xlsx",
            "summary": f"/sync/download/{job_id}/sync_summary.xlsx",
        },
    }
    return JSONResponse(payload)


@app.get("/sync/download/{job_id}/{filename}")
def download_artifact(job_id: str, filename: str):
    path = ARTIFACT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path)


@app.get("/property-configs")
def get_property_configs():
    return PROPERTY_CONFIGS
