import pytest

from twinlab.report_format import number_sections, validate_links, replace_section


def test_section_numbers_update_links_and_survive_regeneration(tmp_path):
    (tmp_path/'docs').mkdir()
    readme = tmp_path/'README.md'
    doc = tmp_path/'docs/03_method.md'
    readme.write_text('[Method](docs/03_method.md#experimental-design)\n')
    doc.write_text('# Method\n\n## Experimental design\n\n### Controls\n\n'
                   '```text\n## Not a heading\n```\n\n## Results\nSaved\n')
    number_sections(tmp_path)
    assert '#31-experimental-design' in readme.read_text()
    assert '### 3.1.1 Controls' in doc.read_text()
    assert '## Not a heading' in doc.read_text()
    before = (readme.read_text(), doc.read_text())
    number_sections(tmp_path)
    assert before == (readme.read_text(), doc.read_text())
    validate_links([readme, doc])
    new = replace_section(doc.read_text(), 'Experimental design', 'Results',
                          '## Experimental design\n\n### Controls\nRebuilt\n')
    doc.write_text(new)
    number_sections(tmp_path)
    validate_links([readme, doc])
    assert '## 3.2 Results\nSaved' in doc.read_text()


def test_publication_rejects_missing_anchor_and_unclosed_fence(tmp_path):
    doc = tmp_path/'README.md'
    doc.write_text('# Title\n\n[Broken](#absent)\n')
    with pytest.raises(ValueError, match='anchor is missing'):
        validate_links([doc])
    doc.write_text('# Title\n\n```math\nx=1\n')
    with pytest.raises(ValueError, match='Unclosed'):
        validate_links([doc])
