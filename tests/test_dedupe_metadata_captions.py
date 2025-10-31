import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dedupe_metadata_captions import clean_record, dedupe_sentences, process_file


def test_dedupe_sentences_preserves_order():
    text = "The woman is posing. The woman is posing. Another sentence."
    assert (
        dedupe_sentences(text)
        == "The woman is posing. Another sentence."
    )


def test_clean_record_preserves_non_caption_fields():
    record = {"file": "text2imagedataset/example.png", "test": "Hello. Hello."}
    cleaned = clean_record(record, "test")

    assert cleaned["file"] == "text2imagedataset/example.png"
    assert cleaned["test"] == "Hello."
    assert record["test"] == "Hello. Hello."


@pytest.mark.parametrize("field", ["text", "test"])
def test_process_file_writes_all_fields(tmp_path: Path, field: str):
    input_path = tmp_path / "metadata.jsonl"
    output_path = tmp_path / "output.jsonl"

    data = {
        "file": "text2imagedataset/image-1.png",
        field: "A caption. A caption."
    }

    input_path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    process_file(input_path, output_path, field)

    out_record = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert out_record["file"] == "text2imagedataset/image-1.png"
    assert out_record[field] == "A caption."
