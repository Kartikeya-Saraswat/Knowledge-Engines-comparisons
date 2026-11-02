
from scripts.link_okf_concepts import (
    normalize_source_name as normalize_link_source_name,
)

from scripts.generate_okf import normalize_source_name


def test_normalize_source_name_accepts_bundle_name():
    assert normalize_source_name("jess101") == "jess101"


def test_normalize_source_name_accepts_pdf_name():
    assert normalize_source_name("jess101.pdf") == "jess101"


def test_normalize_source_name_accepts_json_name():
    assert normalize_source_name("jess101.json") == "jess101"


def test_link_source_name_accepts_bundle_name():
    assert normalize_link_source_name("jess101") == "jess101"


def test_link_source_name_accepts_pdf_name():
    assert normalize_link_source_name("jess101.pdf") == "jess101"


def test_link_source_name_accepts_json_name():
    assert normalize_link_source_name("jess101.json") == "jess101"
