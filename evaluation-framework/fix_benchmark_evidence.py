from pathlib import Path
import json
import shutil
from datetime import datetime

BENCHMARK = Path("benchmark/questions.json")


def question_by_id(questions, qid):
    matches = [q for q in questions if q.get("id") == qid]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {qid}; found {len(matches)}")
    return matches[0]


def anchor_by_evidence(question, evidence_id):
    matches = [
        e for e in question["evidence_units"]
        if e.get("id") == evidence_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{question['id']}: expected exactly one evidence unit {evidence_id}"
        )
    return matches[0]


def main():
    if not BENCHMARK.exists():
        raise SystemExit(
            f"Cannot find {BENCHMARK}. Run this script from evaluation-framework."
        )

    original_text = BENCHMARK.read_text(encoding="utf-8")
    questions = json.loads(original_text)

    if not isinstance(questions, list) or len(questions) != 27:
        raise SystemExit("Expected exactly 27 benchmark questions. No changes made.")

    # Validate all target questions and expected current structure before editing.
    q105_07 = question_by_id(questions, "JESS105-Q07")
    q105_08 = question_by_id(questions, "JESS105-Q08")
    q106_06 = question_by_id(questions, "JESS106-Q06")
    q107_04 = question_by_id(questions, "JESS107-Q04")

    for qid, eid in [
        ("JESS105-Q07", "E1"),
        ("JESS105-Q07", "E2"),
        ("JESS105-Q07", "E3"),
        ("JESS105-Q08", "E2"),
        ("JESS106-Q06", "E3"),
        ("JESS107-Q04", "E1"),
    ]:
        question = question_by_id(questions, qid)
        anchor_by_evidence(question, eid)

    # JESS105-Q07: quote each iron-ore belt from its own exact source passage.
    e1 = anchor_by_evidence(q105_07, "E1")
    e1["pdf_anchors"] = [
        {
            "document": "jess105.pdf",
            "pages": [4],
            "quote": (
                "Odisha-Jharkhand belt: In Odisha high grade hematite ore is "
                "found in Badampahar mines in the Mayurbhanj and Kendujhar "
                "districts. In the adjoining Singbhum district of Jharkhand "
                "haematite iron ore is mined in Gua and Noamundi."
            ),
        },
        {
            "document": "jess105.pdf",
            "pages": [4],
            "quote": (
                "Durg-Bastar-Chandrapur belt lies in Chhattisgarh and "
                "Maharashtra. Very high grade hematites are found in the "
                "famous Bailadila range of hills in the Bastar district of "
                "Chhattisgarh."
            ),
        },
        {
            "document": "jess105.pdf",
            "pages": [4],
            "quote": (
                "Ballari-Chitradurga-Chikkamagaluru-Tumakuru belt in "
                "Karnataka has large reserves of iron ore."
            ),
        },
        {
            "document": "jess105.pdf",
            "pages": [4],
            "quote": (
                "Maharashtra-Goa belt includes the state of Goa and "
                "Ratnagiri district of Maharashtra."
            ),
        },
    ]
    q105_07["expected_answer"] = (
        "The four major iron ore belts are the Odisha-Jharkhand belt, "
        "Durg-Bastar-Chandrapur belt, Ballari-Chitradurga-Chikkamagaluru-"
        "Tumakuru belt in Karnataka, and Maharashtra-Goa belt. The textbook "
        "describes iron ore from the Bailadila mines being exported through "
        "Visakhapatnam, and Kudremukh ore being transported as slurry through "
        "a pipeline to a port near Mangaluru."
    )

    # Both export and slurry-transport descriptions occur on physical PDF page 4.
    e3 = anchor_by_evidence(q105_07, "E3")
    e3["pdf_anchors"][0]["pages"] = [4]

    # JESS105-Q08: the HVJ pipeline description is on physical PDF page 11.
    e2 = anchor_by_evidence(q105_08, "E2")
    e2["pdf_anchors"][0]["pages"] = [11]

    # JESS106-Q06: industrial water-control measures quoted here are on page 9.
    e3 = anchor_by_evidence(q106_06, "E3")
    e3["pdf_anchors"][0]["pages"] = [9]

    # JESS107-Q04: preserve each waterway as an exact, independently checkable quote.
    q107_04["expected_answer"] = (
        "NW-1: Ganga between Prayagraj and Haldia (1,620 km). "
        "NW-2: Brahmaputra between Sadiya and Dhubri (891 km). "
        "NW-3: West-Coast Canal in Kerala, including the Kottapuram-Kollam, "
        "Udyogamandal and Champakkara canals (205 km). "
        "NW-4: specified stretches of the Godavari and Krishna rivers, "
        "along with the Kakinada-Puducherry canal stretch (1,078 km). "
        "NW-5: specified stretches of the Brahmani river with the Matai "
        "river, the delta channels of the Mahanadi and Brahmani rivers, "
        "and the East Coast Canal (588 km)."
    )
    e1 = anchor_by_evidence(q107_04, "E1")
    e1["pdf_anchors"] = [
        {
            "document": "jess107.pdf",
            "pages": [7],
            "quote": "The Ganga river between Prayagraj and Haldia (1620 km)-N.W. No.1",
        },
        {
            "document": "jess107.pdf",
            "pages": [7],
            "quote": "The Brahmaputra river between Sadiya and Dhubri (891 km)-N.W. No.2",
        },
        {
            "document": "jess107.pdf",
            "pages": [7],
            "quote": (
                "The West-Coast Canal in Kerala "
                "(Kottapurma-Kollam, Udyogamandal and "
                "Champakkara canals-205 km) – N.W. No.3"
            ),
        },
        {
            "document": "jess107.pdf",
            "pages": [7],
            "quote": (
                "Specified stretches of Godavari and Krishna rivers along "
                "with Kakinada Puducherry stretch of canals (1078 km) – N.W. No.4"
            ),
        },
        {
            "document": "jess107.pdf",
            "pages": [7],
            "quote": (
                "Specified stretches of river Brahmani along with Matai "
                "river, delta channels of Mahanadi and Brahmani rivers and "
                "East Coast Canal (588 km) – N.W. No.5"
            ),
        },
    ]

    # Preserve a timestamped backup before writing.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BENCHMARK.with_name(f"questions.json.before_evidence_fix_{timestamp}.bak")
    shutil.copy2(BENCHMARK, backup)

    BENCHMARK.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Re-read and perform basic post-write checks.
    saved = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if len(saved) != 27:
        raise SystemExit("Post-write validation failed: question count changed.")
    if len({q["id"] for q in saved}) != 27:
        raise SystemExit("Post-write validation failed: duplicate IDs found.")

    print("Evidence annotation corrections applied.")
    print(f"Questions retained: {len(saved)}")
    print(f"Backup created: {backup}")
    print("Updated: JESS105-Q07, JESS105-Q08, JESS106-Q06, JESS107-Q04")


if __name__ == "__main__":
    main()
