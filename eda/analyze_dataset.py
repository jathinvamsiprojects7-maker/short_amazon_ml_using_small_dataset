"""Streaming EDA for the Amazon ML business entity resolution data."""

from __future__ import annotations

import csv
import difflib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "student_resource" / "dataset"
OUT = ROOT / "eda"
CHUNK_SAMPLE = 5000
PAIR_SAMPLE_LIMIT = 20000
RNG = Random(20260925)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        for line_number, row in enumerate(reader, 2):
            yield line_number, header, row


def clean(value: str) -> str:
    return value.strip()


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def norm_compact(value: str) -> str:
    return re.sub(r"[^\w]", "", norm_text(value), flags=re.UNICODE)


def tokens(value: str) -> list[str]:
    return norm_text(value).split()


def percent(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def quantiles(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {}
    values = sorted(values)
    result: dict[str, float | int] = {"min": values[0], "max": values[-1]}
    for label, q in (("p01", 0.01), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p99", 0.99)):
        result[label] = values[min(len(values) - 1, int(q * (len(values) - 1)))]
    result["mean"] = round(statistics.fmean(values), 4)
    return result


class Reservoir:
    def __init__(self, limit: int = CHUNK_SAMPLE):
        self.limit = limit
        self.items: list = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(item)
        else:
            index = RNG.randrange(self.seen)
            if index < self.limit:
                self.items[index] = item


def empty_field(value: str) -> bool:
    return not value.strip()


def row_stats(path: Path) -> dict:
    counts = Counter()
    expected = ["entity_id", "business_name", "business_address", "country"]
    value_freq = {field: Counter() for field in expected}
    samples = {field: Reservoir(100) for field in expected}
    lengths = {field: Reservoir(10000) for field in ("business_name", "business_address")}
    token_lengths = {field: Reservoir(10000) for field in ("business_name", "business_address")}
    malformed = 0
    header_seen = None
    for _, header, row in read_rows(path):
        header_seen = header
        if header != expected or len(row) != len(expected):
            malformed += 1
            continue
        record = dict(zip(expected, row))
        counts["rows"] += 1
        for field, value in record.items():
            value = clean(value)
            if empty_field(value):
                counts[f"empty_{field}"] += 1
            else:
                value_freq[field][value] += 1
                samples[field].add(value)
            if field in lengths:
                lengths[field].add(len(value))
                token_lengths[field].add(len(tokens(value)))
    duplicate_counts = {
        field: sum(freq - 1 for freq in freqs.values() if freq > 1)
        for field, freqs in value_freq.items()
    }
    duplicate_value_counts = {
        field: sum(1 for freq in freqs.values() if freq > 1)
        for field, freqs in value_freq.items()
    }
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "header": header_seen,
        "expected_header": expected,
        "delimiter_valid": header_seen == expected,
        "rows": counts["rows"],
        "malformed_rows": malformed,
        "empty_counts": {field: counts[f"empty_{field}"] for field in expected},
        "missing_percent": {field: percent(counts[f"empty_{field}"], counts["rows"]) for field in expected},
        "unique_counts": {field: len(value_freq[field]) for field in expected},
        "duplicate_row_counts": duplicate_counts,
        "duplicate_value_counts": duplicate_value_counts,
        "duplicate_percent": {field: percent(duplicate_counts[field], counts["rows"]) for field in expected},
        "lengths": {field: quantiles(lengths[field].items) for field in lengths},
        "token_counts": {field: quantiles(token_lengths[field].items) for field in token_lengths},
        "top_values": {field: value_freq[field].most_common(20) for field in ("business_name", "business_address", "country")},
        "sample_values": {field: samples[field].items[:20] for field in samples},
    }


def source_features(stats: dict, path: Path) -> dict:
    name_freq = Counter()
    address_freq = Counter()
    country_freq = Counter()
    norm_name_freq = Counter()
    norm_address_freq = Counter()
    name_lengths = Counter()
    address_lengths = Counter()
    name_tokens = Counter()
    address_tokens = Counter()
    patterns = Counter()
    examples = Reservoir(100)
    for _, _, row in read_rows(path):
        if len(row) != 4:
            continue
        entity_id, name, address, country = map(clean, row)
        name_freq[name] += 1
        address_freq[address] += 1
        country_freq[country] += 1
        normalized_name = norm_text(name)
        normalized_address = norm_text(address)
        norm_name_freq[normalized_name] += 1
        norm_address_freq[normalized_address] += 1
        name_lengths[len(name)] += 1
        address_lengths[len(address)] += 1
        name_tokens.update(tokens(name))
        address_tokens.update(tokens(address))
        patterns.update({
            "name_has_ampersand": "&" in name,
            "name_has_and": bool(re.search(r"\band\b", name, re.I)),
            "name_has_digit": bool(re.search(r"\d", name)),
            "name_has_punctuation": bool(re.search(r"[^\w\s]", name, re.UNICODE)),
            "name_has_non_ascii": any(ord(char) > 127 for char in name),
            "address_has_digit": bool(re.search(r"\d", address)),
            "address_has_punctuation": bool(re.search(r"[^\w\s]", address, re.UNICODE)),
            "address_has_non_ascii": any(ord(char) > 127 for char in address),
        })
        examples.add({"id": entity_id, "name": name, "address": address, "country": country})
    def repeated(counter: Counter) -> dict:
        return {"values_repeated": sum(1 for n in counter.values() if n > 1), "rows_in_repeated_values": sum(n for n in counter.values() if n > 1), "top": counter.most_common(20)}
    stats.update({
        "countries": country_freq.most_common(),
        "business_name": {"unique": len(name_freq), "repeated": repeated(name_freq), "normalized_unique": len(norm_name_freq), "normalized_repeated": repeated(norm_name_freq), "length_counts": name_lengths.most_common(), "token_counts": name_tokens.most_common(30), "patterns": dict(patterns)},
        "business_address": {"unique": len(address_freq), "repeated": repeated(address_freq), "normalized_unique": len(norm_address_freq), "normalized_repeated": repeated(norm_address_freq), "length_counts": address_lengths.most_common(), "token_counts": address_tokens.most_common(40)},
        "examples": examples.items,
    })
    return stats


def load_rows_for_ids(path: Path, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    rows = {}
    for _, _, row in read_rows(path):
        if len(row) == 4 and row[0] in wanted:
            rows[row[0]] = tuple(clean(value) for value in row[1:])
    return rows


def read_ground_truth(path: Path) -> tuple[dict[str, list[str]], Counter, list[dict]]:
    links: dict[str, list[str]] = {}
    s2_inverse = Counter()
    s3_inverse = Counter()
    examples = Reservoir(100)
    malformed = 0
    for _, header, row in read_rows(path):
        if header != ["source1_entity_id", "matched_entity_ids"] or len(row) != 2:
            malformed += 1
            continue
        s1, raw_ids = clean(row[0]), clean(row[1])
        matched = [item.strip() for item in raw_ids.split(",") if item.strip()]
        links[s1] = matched
        for entity_id in matched:
            (s2_inverse if entity_id.startswith("S2-") else s3_inverse)[entity_id] += 1
        examples.add({"source1": s1, "matches": matched})
    return links, Counter({"malformed": malformed, "s1_entities": len(links), "zero": sum(not x for x in links.values()), "one": sum(len(x) == 1 for x in links.values()), "multiple": sum(len(x) > 1 for x in links.values()), "total_links": sum(map(len, links.values())), "max_matches": max(map(len, links.values()), default=0), "s2_links": sum(x.startswith("S2-") for values in links.values() for x in values), "s3_links": sum(x.startswith("S3-") for values in links.values() for x in values)}), [*examples.items, {"s2_inverse": s2_inverse, "s3_inverse": s3_inverse}]


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def true_pair_analysis(links: dict[str, list[str]], sources: dict[str, dict], sample_limit: int = PAIR_SAMPLE_LIMIT) -> dict:
    pair_sample = Reservoir(sample_limit)
    for s1, matched_ids in links.items():
        left = sources["source1"].get(s1)
        if not left:
            continue
        for matched_id in matched_ids:
            source = "source2" if matched_id.startswith("S2-") else "source3"
            right = sources[source].get(matched_id)
            if right:
                pair_sample.add((left, right, matched_id))
    rates = Counter()
    name_sims: list[float] = []
    address_sims: list[float] = []
    missing = Counter()
    relation = Counter()
    for left, right, _ in pair_sample.items:
        left_name, left_address, left_country = left
        right_name, right_address, right_country = right
        left_norm_name, right_norm_name = norm_text(left_name), norm_text(right_name)
        left_norm_address, right_norm_address = norm_text(left_address), norm_text(right_address)
        rates["raw_name_exact"] += bool(left_name == right_name and left_name)
        rates["normalized_name_exact"] += bool(left_norm_name == right_norm_name and left_norm_name)
        rates["raw_address_exact"] += bool(left_address == right_address and left_address)
        rates["normalized_address_exact"] += bool(left_norm_address == right_norm_address and left_norm_address)
        rates["same_country"] += bool(left_country and left_country.casefold() == right_country.casefold())
        rates["cross_country"] += bool(left_country and right_country and left_country.casefold() != right_country.casefold())
        name_sims.append(similarity(left_norm_name, right_norm_name))
        address_sims.append(similarity(left_norm_address, right_norm_address))
        missing["left_name"] += empty_field(left_name)
        missing["right_name"] += empty_field(right_name)
        missing["left_address"] += empty_field(left_address)
        missing["right_address"] += empty_field(right_address)
        if left_norm_name == right_norm_name and left_norm_name and left_norm_address == right_norm_address and left_norm_address:
            relation["name_and_address"] += 1
        elif left_norm_name == right_norm_name and left_norm_name:
            relation["name_only"] += 1
        elif left_norm_address == right_norm_address and left_norm_address:
            relation["address_only"] += 1
        else:
            relation["neither_exact"] += 1
    denominator = len(pair_sample.items)
    return {"sampled_pairs": denominator, "sample_seen": pair_sample.seen, "sampling_fraction": percent(denominator, pair_sample.seen), "rates_percent": {key: percent(value, denominator) for key, value in rates.items()}, "similarity": {"name": quantiles(name_sims), "address": quantiles(address_sims)}, "missing_percent": {key: percent(value, denominator) for key, value in missing.items()}, "exact_relation_percent": {key: percent(value, denominator) for key, value in relation.items()}}


def collision_analysis(source1: dict, source2: dict, source3: dict, links: dict[str, list[str]]) -> dict:
    sources = {"S2": source2["rows_by_id"], "S3": source3["rows_by_id"]}
    indexes = {}
    for prefix, rows in sources.items():
        indexes[prefix] = {
            "name": defaultdict(list), "address": defaultdict(list), "country_name": defaultdict(list), "name_tokens": defaultdict(list), "address_tokens": defaultdict(list), "address_digits": defaultdict(list)
        }
        for entity_id, (name, address, country) in rows.items():
            index = indexes[prefix]
            index["name"][norm_text(name)].append(entity_id)
            index["address"][norm_text(address)].append(entity_id)
            index["country_name"][(country.casefold(), norm_text(name))].append(entity_id)
            for token in set(tokens(name)):
                if len(token) >= 3:
                    index["name_tokens"][token].append(entity_id)
            for token in set(tokens(address)):
                if len(token) >= 3:
                    index["address_tokens"][token].append(entity_id)
            digits = " ".join(re.findall(r"\d+", address))
            if digits:
                index["address_digits"][digits].append(entity_id)
    candidate_totals = Counter()
    true_covered = Counter()
    collision_sizes = {key: [] for key in ("name", "address", "country_name", "name_tokens", "address_tokens", "address_digits")}
    for s1_id, matched in links.items():
        left = source1["rows_by_id"].get(s1_id)
        if not left:
            continue
        name, address, country = left
        true_set = set(matched)
        for prefix in ("S2", "S3"):
            index = indexes[prefix]
            keys = {"name": norm_text(name), "address": norm_text(address), "country_name": (country.casefold(), norm_text(name)), "name_tokens": set(tokens(name)), "address_tokens": set(tokens(address)), "address_digits": " ".join(re.findall(r"\d+", address))}
            for signal, key in keys.items():
                if signal in ("name_tokens", "address_tokens"):
                    candidates = set(entity for token in key if len(token) >= 3 for entity in index[signal].get(token, []))
                else:
                    candidates = set(index[signal].get(key, [])) if key else set()
                candidate_totals[signal] += len(candidates)
                collision_sizes[signal].append(len(candidates))
                true_covered[signal] += bool(true_set.intersection(candidates))
    pair_count = sum(1 for values in links.values() for _ in values)
    s1_count = len(links) * 2
    return {"known_positive_pair_count": pair_count, "candidate_volume_mean_per_s1_source": {key: round(value / s1_count, 4) for key, value in candidate_totals.items()}, "positive_coverage_percent": {key: percent(value, s1_count) for key, value in true_covered.items()}, "candidate_set_size_quantiles": {key: quantiles(values) for key, values in collision_sizes.items()}, "dangerous_name_collisions": {prefix: indexes[prefix]["name"].get(norm_text(name), []) for prefix in () for name in ()}}


def shift_summary(train_stats: dict, test_stats: dict) -> dict:
    result = {}
    for key in ("rows", "unique_counts", "missing_percent", "lengths", "token_counts", "countries"):
        result[key] = {"train": train_stats.get(key), "test": test_stats.get(key)}
    return result


def compact_stats(stats: dict) -> dict:
    stats = dict(stats)
    stats.pop("rows_by_id", None)
    return stats


def make_markdown(profile: dict) -> str:
    inv = profile["inventory"]
    gt = profile["ground_truth"]
    lines = ["# Dataset Profile", "", "Generated by `eda/analyze_dataset.py` with streaming reads. Original TSV files were not modified.", "", "## Inventory", "", "| File | Rows | Size (MB) | Malformed | Missing fields |", "|---|---:|---:|---:|---|"]
    for item in inv:
        missing = ", ".join(f"{k}: {v:.2f}%" for k, v in item["missing_percent"].items() if v)
        lines.append(f"| `{item['path']}` | {item['rows']:,} | {item['size_bytes'] / 1_000_000:.2f} | {item['malformed_rows']:,} | {missing or 'none'} |")
    lines += ["", "## Training Match Distribution", "", f"- S1 entities: **{gt['s1_entities']:,}**; zero: **{gt['zero']:,}** ({percent(gt['zero'], gt['s1_entities']):.2f}%); exactly one: **{gt['one']:,}**; multiple: **{gt['multiple']:,}**.", f"- Total links: **{gt['total_links']:,}**; maximum matches for one S1: **{gt['max_matches']:,}**; S2 links: **{gt['s2_links']:,}**; S3 links: **{gt['s3_links']:,}**.", "", "## Source Characteristics", "", "| Source | Country distribution | Common exact name | Normalized-name collision rows | Address collision rows |", "|---|---|---|---:|---:|"]
    for source_name in ("train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"):
        source = profile["sources"][source_name]
        top_name = source["business_name"]["repeated"]["top"][0] if source["business_name"]["repeated"]["top"] else ("", 0)
        lines.append(f"| {source_name} | {', '.join(f'{c}: {n:,}' for c, n in source['countries'][:5])} | {top_name[0]!r} ({top_name[1]:,}) | {source['business_name']['normalized_repeated']['rows_in_repeated_values']:,} | {source['business_address']['repeated']['rows_in_repeated_values']:,} |")
    pair = profile["true_pair_analysis"]
    lines += ["", "## True-Pair Characteristics", "", f"Pair analysis sampled **{pair['sampled_pairs']:,}** of **{pair['sample_seen']:,}** known links using a fixed reservoir (seed 20260925).", "", "| Signal | Rate |", "|---|---:|"]
    for key, value in pair["rates_percent"].items():
        lines.append(f"| {key} | {value:.2f}% |")
    lines += ["", "## Blocking Signal Diagnostics", "", "| Signal | Positive coverage | Mean candidates per S1/source |", "|---|---:|---:|"]
    for key in profile["blocking"]["positive_coverage_percent"]:
        lines.append(f"| {key} | {profile['blocking']['positive_coverage_percent'][key]:.2f}% | {profile['blocking']['candidate_volume_mean_per_s1_source'].get(key, 0):.2f} |")
    lines += ["", "## Train/Test Shift", "", "Country labels are reported directly from test; France is not inferred from training.", "", "```json", json.dumps(profile["shift"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    train = DATASET / "train"
    test = DATASET / "test"
    paths = {
        "train_source1": train / "train_source1.tsv", "train_source2": train / "train_source2.tsv", "train_source3": train / "train_source3.tsv", "train_ground_truth": train / "train_ground_truth.tsv",
        "test_source1": test / "test_source1.tsv", "test_source2": test / "test_source2.tsv", "test_source3": test / "test_source3.tsv",
    }
    profile: dict = {"analysis": {"seed": 20260925, "pair_sample_limit": PAIR_SAMPLE_LIMIT, "normalization": "NFKC, casefold, ampersand to and, punctuation to spaces, whitespace collapse"}, "inventory": [], "sources": {}}
    links, gt_counts, gt_examples = read_ground_truth(paths["train_ground_truth"])
    sampled_pairs = Reservoir(PAIR_SAMPLE_LIMIT)
    for s1, matched_ids in links.items():
        for matched_id in matched_ids:
            sampled_pairs.add((s1, matched_id))
    wanted_ids = {"source1": {s1 for s1, _ in sampled_pairs.items}, "source2": set(), "source3": set()}
    for _, matched_id in sampled_pairs.items:
        wanted_ids["source2" if matched_id.startswith("S2-") else "source3"].add(matched_id)
    source_data = {}
    for key, path in paths.items():
        if key == "train_ground_truth":
            continue
        stats = row_stats(path)
        stats = source_features(stats, path)
        profile["inventory"].append(compact_stats(stats))
        profile["sources"][key] = stats
        if key.startswith("train_"):
            source_name = key.replace("train_", "")
            source_data[source_name] = {"rows_by_id": load_rows_for_ids(path, wanted_ids[source_name])}
    profile["inventory"].append({"path": str(paths["train_ground_truth"].relative_to(ROOT)), "size_bytes": paths["train_ground_truth"].stat().st_size, "rows": gt_counts["s1_entities"], "malformed_rows": gt_counts["malformed"], "header": ["source1_entity_id", "matched_entity_ids"], "delimiter_valid": True})
    s2_inverse = Counter(entity_id for values in links.values() for entity_id in values if entity_id.startswith("S2-"))
    s3_inverse = Counter(entity_id for values in links.values() for entity_id in values if entity_id.startswith("S3-"))
    profile["ground_truth"] = {**gt_counts, "match_count_distribution": Counter(map(len, links.values())), "inverse_uniqueness": {"s2_unique_to_one_s1": sum(n == 1 for n in s2_inverse.values()), "s2_to_multiple_s1": sum(n > 1 for n in s2_inverse.values()), "s3_unique_to_one_s1": sum(n == 1 for n in s3_inverse.values()), "s3_to_multiple_s1": sum(n > 1 for n in s3_inverse.values()), "s2_distinct_ids": len(s2_inverse), "s3_distinct_ids": len(s3_inverse)}}
    profile["true_pair_analysis"] = true_pair_analysis(links, source_data)
    profile["blocking"] = collision_analysis(source_data["source1"], source_data["source2"], source_data["source3"], links)
    profile["shift"] = shift_summary(profile["sources"]["train_source1"], profile["sources"]["test_source1"])
    profile["test_countries"] = {key: profile["sources"][key]["countries"] for key in ("test_source1", "test_source2", "test_source3")}
    examples = {"ground_truth": gt_examples[:20]}
    for key in profile["sources"]:
        examples[key] = profile["sources"][key]["examples"][:20]
    OUT.mkdir(exist_ok=True)
    serializable = json.loads(json.dumps(profile, default=lambda value: dict(value) if isinstance(value, Counter) else value, ensure_ascii=False))
    (OUT / "dataset_profile.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "dataset_profile.md").write_text(make_markdown(serializable), encoding="utf-8")
    (OUT / "important_examples.txt").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files_profiled": len(paths), "s1_train": gt_counts["s1_entities"], "total_links": gt_counts["total_links"], "pair_sampled": profile["true_pair_analysis"]["sampled_pairs"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()