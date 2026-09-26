"""Audit full-persona answers without opening reserved participants' JSON."""
import json

from .data import target_atoms


def audit_full_personas(rows, references, membership):
    """Compare every expected response, including unchanged and missing answers.

    references maps participant ID to response key to (earlier Atom, later Atom).
    It must cover exactly the training/development participants. Missing source
    participants and responses remain in the denominator, never silently dropped.
    """
    expected = {pid for pid, split in membership.items() if split in {"train", "dev"}}
    if set(references) != expected:
        raise ValueError("Audit references must cover exactly train/dev participants")
    stats = dict.fromkeys([
        "participants", "missing_participants", "missing_responses",
        "present_responses", "invalid_responses", "missing_or_invalid_values",
        "reference_missing_values", "stimulus_mismatches", "comparable_responses",
        "matches_wave4", "disagrees_with_wave4", "changed_responses",
        "unchanged_responses", "changed_atom_comparisons", "changed_matches_wave4",
        "changed_disagrees_with_wave4", "unchanged_atom_comparisons",
        "unchanged_matches_wave4", "unchanged_disagrees_with_wave4",
        "unexpected_response_keys",
    ], 0)
    stats["expected_participants"] = len(expected)
    stats["expected_responses"] = sum(map(len, references.values()))
    for items in references.values():
        for early, late in items.values():
            if early.value is None or late.value is None:
                stats["reference_missing_values"] += 1
            else:
                stats["changed_responses" if early.value != late.value else "unchanged_responses"] += 1
    seen = set()
    for row in rows:
        pid = str(row["pid"])
        if pid not in membership:
            raise ValueError(f"Unknown full-persona participant: {pid}")
        if membership[pid] == "test":
            continue  # Do not deserialize reserved participants' full personas.
        if pid in seen:
            raise ValueError(f"Duplicate full-persona participant: {pid}")
        seen.add(pid)
        items = references[pid]
        questions = {(late.condition, late.qid) for _, late in items.values()}
        selected = []
        for block in json.loads(row["persona_json"]):
            keep = [q for q in block.get("Questions", [])
                    if (block.get("BlockName", "").strip(), q["QuestionID"]) in questions]
            if keep:
                selected.append({**block, "Questions": keep})
        actual = {atom.key: atom for atom in target_atoms(selected)}
        stats["unexpected_response_keys"] += len(actual.keys() - items.keys())
        for key, (early, late) in items.items():
            if key not in actual:
                stats["missing_responses"] += 1
                continue
            atom = actual[key]
            stats["present_responses"] += 1
            stats["invalid_responses"] += int(atom.invalid)
            stats["missing_or_invalid_values"] += int(atom.value is None)
            stats["stimulus_mismatches"] += int(atom.stimulus_hash != late.stimulus_hash)
            if atom.value is None or late.value is None or atom.stimulus_hash != late.stimulus_hash:
                continue
            matches = atom.value == late.value
            stats["comparable_responses"] += 1
            stats["matches_wave4" if matches else "disagrees_with_wave4"] += 1
            if early.value is not None:
                group = "changed" if early.value != late.value else "unchanged"
                stats[f"{group}_atom_comparisons"] += 1
                stats[f"{group}_matches_wave4" if matches else f"{group}_disagrees_with_wave4"] += 1
    stats["participants"] = len(seen)
    stats["missing_participants"] = len(expected - seen)
    stats["missing_responses"] += sum(len(references[pid]) for pid in expected - seen)
    stats["reserved_test_participants_excluded"] = sum(s == "test" for s in membership.values())
    return stats
