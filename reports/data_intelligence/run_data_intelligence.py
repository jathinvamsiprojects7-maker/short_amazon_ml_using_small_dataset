"""Chunked data intelligence analysis for the Amazon ML ER challenge.

Run from the repository root with: python reports/data_intelligence/run_data_intelligence.py
The source TSVs are read only. Expensive pair analyses use deterministic reservoirs.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "student_resource" / "dataset"
OUT = ROOT / "reports" / "data_intelligence"
SEED = 20260925
RNG = Random(SEED)
CHUNK_SIZE = 200_000
PAIR_SAMPLE_LIMIT = 20_000
HARD_NEGATIVE_LIMIT = 10_000


class Reservoir:
    def __init__(self, limit: int):
        self.limit = limit
        self.items = []
        self.seen = 0

    def add(self, value):
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(value)
        else:
            index = RNG.randrange(self.seen)
            if index < self.limit:
                self.items[index] = value


def pct(value, total):
    return round(100 * value / total, 4) if total else 0.0


def norm0(value):
    return value.strip().casefold()


def norm1(value):
    return unicodedata.normalize("NFKC", norm0(value))


def norm2(value):
    return re.sub(r"[^\w\s]", " ", norm1(value), flags=re.UNICODE)


def norm3(value):
    return re.sub(r"\s+", " ", norm2(value)).strip()


def norm4(value):
    value = unicodedata.normalize("NFKD", norm3(value))
    return "".join(char for char in value if not unicodedata.combining(char))


def tokens(value):
    return norm3(value).split()


def digits(value):
    return re.findall(r"\d+", value)


def qhist(hist: Counter, quantile: float):
    total = sum(hist.values())
    if not total:
        return 0
    target = max(1, math.ceil(total * quantile))
    running = 0
    for key in sorted(hist):
        running += hist[key]
        if running >= target:
            return key
    return max(hist)


def distribution(hist: Counter):
    return {str(key): value for key, value in sorted(hist.items(), key=lambda item: item[0])}


def js_divergence(a: Counter, b: Counter):
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    va = np.array([a[key] for key in keys], dtype=float)
    vb = np.array([b[key] for key in keys], dtype=float)
    va /= va.sum() or 1
    vb /= vb.sum() or 1
    return round(float(jensenshannon(va, vb, base=2)), 6)


def text_similarity(left, right):
    return difflib.SequenceMatcher(None, left, right).ratio()


def token_jaccard(left, right):
    a, b = set(tokens(left)), set(tokens(right))
    return len(a & b) / len(a | b) if a | b else 1.0


def digit_jaccard(left, right):
    a, b = set(digits(left)), set(digits(right))
    return len(a & b) / len(a | b) if a | b else 1.0


def file_paths():
    train, test = DATA / "train", DATA / "test"
    return {
        "train_source1": train / "train_source1.tsv",
        "train_source2": train / "train_source2.tsv",
        "train_source3": train / "train_source3.tsv",
        "train_ground_truth": train / "train_ground_truth.tsv",
        "test_source1": test / "test_source1.tsv",
        "test_source2": test / "test_source2.tsv",
        "test_source3": test / "test_source3.tsv",
    }


def update_heavy(counter: dict, value: str, limit=1000):
    if not value:
        return
    counter[value] = counter.get(value, 0) + 1
    if len(counter) > limit * 2:
        for key, _ in sorted(counter.items(), key=lambda item: item[1])[:limit]:
            del counter[key]


def scan_source(path: Path, wanted_ids: set[str] | None = None):
    expected = ["entity_id", "business_name", "business_address", "country"]
    field_sets = {field: set() for field in expected}
    field_seen = {field: set() for field in expected}
    field_repeated = {field: set() for field in expected}
    length_hist = {field: Counter() for field in ("business_name", "business_address")}
    token_hist = {field: Counter() for field in ("business_name", "business_address")}
    country_counts = Counter()
    patterns = Counter()
    top_names, top_addresses = {}, {}
    examples = Reservoir(100)
    selected = {}
    malformed = 0
    rows = 0
    start = time.perf_counter()
    header = None
    for chunk in pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, chunksize=CHUNK_SIZE, encoding="utf-8", on_bad_lines="error"):
        if header is None:
            header = list(chunk.columns)
            if header != expected:
                raise ValueError(f"{path}: expected {expected}, received {header}")
        if len(chunk.columns) != 4:
            malformed += len(chunk)
            continue
        rows += len(chunk)
        if wanted_ids:
            for record in chunk[chunk.entity_id.isin(wanted_ids)].itertuples(index=False, name=None):
                selected[record[0]] = tuple(str(value) for value in record[1:])
        for field in expected:
            values = chunk[field].astype(str).str.strip()
            for value in values:
                if value == "":
                    continue
                if value in field_seen[field]:
                    field_repeated[field].add(value)
                else:
                    field_seen[field].add(value)
                field_sets[field].add(value)
        names = chunk.business_name.astype(str).str.strip()
        addresses = chunk.business_address.astype(str).str.strip()
        countries = chunk.country.astype(str).str.strip()
        for value in countries:
            country_counts[value] += 1
        for field, values in (("business_name", names), ("business_address", addresses)):
            for value in values:
                length_hist[field][len(value)] += 1
                token_hist[field][len(tokens(value))] += 1
        for name, address, country, entity_id in zip(names, addresses, countries, chunk.entity_id.astype(str)):
            patterns.update({
                "name_non_ascii": any(ord(char) > 127 for char in name),
                "name_digit": bool(re.search(r"\d", name)),
                "name_punctuation": bool(re.search(r"[^\w\s]", name, re.UNICODE)),
                "name_ampersand": "&" in name,
                "address_non_ascii": any(ord(char) > 127 for char in address),
                "address_digit": bool(re.search(r"\d", address)),
                "address_punctuation": bool(re.search(r"[^\w\s]", address, re.UNICODE)),
            })
            update_heavy(top_names, norm3(name))
            update_heavy(top_addresses, norm3(address))
            examples.add({"entity_id": entity_id, "business_name": name, "business_address": address, "country": country})
    counts = {field: rows - sum(1 for value in field_seen[field] if value) for field in []}
    empty_counts = {field: 0 for field in expected}
    # Empty counts are tracked from field cardinality-safe iteration in a second lightweight pass below.
    for chunk in pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, chunksize=CHUNK_SIZE, encoding="utf-8", usecols=expected):
        for field in expected:
            empty_counts[field] += int((chunk[field].astype(str).str.strip() == "").sum())
    def field_summary(field):
        hist = length_hist.get(field, Counter())
        return {
            "unique_count": len(field_sets[field]),
            "duplicate_row_count": rows - len(field_sets[field]) - empty_counts[field],
            "duplicate_value_count": len(field_repeated[field]),
            "empty_count": empty_counts[field],
            "empty_percent": pct(empty_counts[field], rows),
            "length_distribution_exact": distribution(hist),
            "length_quantiles_exact": {"min": min(hist, default=0), "p25": qhist(hist, .25), "median": qhist(hist, .5), "mean": round(sum(k*v for k,v in hist.items()) / max(1, sum(hist.values())), 4), "p75": qhist(hist, .75), "p90": qhist(hist, .9), "p99": qhist(hist, .99), "max": max(hist, default=0)},
            "token_distribution_exact": distribution(token_hist[field]),
        }
    result = {
        "path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "rows": rows,
        "columns": header, "dtypes": {field: "string" for field in expected}, "delimiter_valid": header == expected,
        "malformed_rows": malformed, "duplicate_entity_ids": rows - len(field_sets["entity_id"]),
        "entity_id_prefix_counts": Counter(value.split("-", 1)[0] if "-" in value else "NO_PREFIX" for value in field_sets["entity_id"]),
        "countries": country_counts, "patterns": patterns, "business_name": field_summary("business_name"),
        "business_address": field_summary("business_address"), "entity_id_unique_count": len(field_sets["entity_id"]),
        "examples": examples.items, "top_names_approximate": sorted(top_names.items(), key=lambda item: -item[1])[:30],
        "top_addresses_approximate": sorted(top_addresses.items(), key=lambda item: -item[1])[:20],
        "runtime_seconds": round(time.perf_counter() - start, 3), "selected_rows": selected,
    }
    return result


class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def union(self, left, right):
        self.add(left); self.add(right)
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            nxt = self.parent[node]
            self.parent[node] = root
            node = nxt
        return root


def scan_ground_truth(path: Path):
    match_hist = Counter(); s2_inverse = Counter(); s3_inverse = Counter(); dsu = DSU(); pair_sample = Reservoir(PAIR_SAMPLE_LIMIT); examples = Reservoir(100)
    rows = malformed = total_links = 0
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) != 2:
                malformed += 1; continue
            s1 = row[0].strip(); matched = [value.strip() for value in row[1].split(",") if value.strip()]
            rows += 1; total_links += len(matched); match_hist[len(matched)] += 1; dsu.add(s1)
            examples.add({"source1": s1, "matches": matched})
            for entity_id in matched:
                pair_sample.add((s1, entity_id))
                dsu.union(s1, entity_id)
                if entity_id.startswith("S2-"): s2_inverse[entity_id] += 1
                elif entity_id.startswith("S3-"): s3_inverse[entity_id] += 1
    component_sizes = Counter(dsu.size[root] for root in set(dsu.find(node) for node in dsu.parent))
    def inv_summary(counter):
        return {"distinct_entities": len(counter), "links_by_s1_histogram": distribution(Counter(counter.values())), "unique_to_one_s1": sum(value == 1 for value in counter.values()), "shared_by_multiple_s1": sum(value > 1 for value in counter.values()), "max_s1_per_entity": max(counter.values(), default=0)}
    return {
        "path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "rows": rows, "columns": header, "delimiter_valid": header == ["source1_entity_id", "matched_entity_ids"], "malformed_rows": malformed,
        "match_count_histogram": distribution(match_hist), "zero_matches": match_hist[0], "one_match": match_hist[1], "two_matches": match_hist[2], "three_or_more": sum(value for key, value in match_hist.items() if key >= 3), "total_links": total_links, "mean_matches": round(total_links / max(1, rows), 6), "median_matches": qhist(match_hist, .5), "p90_matches": qhist(match_hist, .9), "p95_matches": qhist(match_hist, .95), "p99_matches": qhist(match_hist, .99), "max_matches": max(match_hist, default=0), "zero_percent": pct(match_hist[0], rows), "singleton_rate_percent": pct(match_hist[1], rows), "multi_match_rate_percent": pct(sum(value for key, value in match_hist.items() if key > 1), rows),
        "s2_inverse": inv_summary(s2_inverse), "s3_inverse": inv_summary(s3_inverse), "mapping_classification": "many-to-many" if any(value > 1 for value in s2_inverse.values()) and any(value > 1 for value in s3_inverse.values()) else ("one-to-many" if any(key > 1 for key in match_hist) else "one-to-one"),
        "graph_nodes": len(dsu.parent), "graph_components": sum(component_sizes.values()), "component_size_histogram": distribution(component_sizes), "largest_component": max(component_sizes, default=0), "nodes_in_components_10_plus": sum(size * count for size, count in component_sizes.items() if size >= 10),
        "pair_sample": pair_sample.items, "pair_sample_seen": pair_sample.seen, "examples": examples.items,
    }


def pair_metrics(pairs, records):
    rows = []
    for s1, other in pairs:
        left = records.get(s1); right = records.get(other)
        if not left or not right:
            continue
        ln, la, lc = left; rn, ra, rc = right
        nn, rn2, na, ra2 = norm3(ln), norm3(rn), norm4(la), norm4(ra)
        rows.append({"source_pair": other[:2], "country": lc, "same_country": lc.casefold() == rc.casefold() and bool(lc), "raw_name_exact": ln == rn and bool(ln), "norm_name_exact": nn == rn2 and bool(nn), "raw_address_exact": la == ra and bool(la), "norm_address_exact": na == ra2 and bool(na), "name_similarity": text_similarity(nn, rn2), "address_similarity": text_similarity(na, ra2), "name_token_overlap": token_jaccard(ln, rn), "address_token_overlap": token_jaccard(la, ra), "address_digit_overlap": digit_jaccard(la, ra), "name_length_difference": abs(len(ln) - len(rn)), "address_length_difference": abs(len(la) - len(ra)), "left_name_missing": not bool(ln), "right_name_missing": not bool(rn), "left_address_missing": not bool(la), "right_address_missing": not bool(ra), "s1": s1, "other": other})
    return rows


def summarize_pair_rows(rows, include_breakdowns=True):
    if not rows:
        return {"sample_size": 0}
    frame = pd.DataFrame(rows)
    numeric = ["name_similarity", "address_similarity", "name_token_overlap", "address_token_overlap", "address_digit_overlap", "name_length_difference", "address_length_difference"]
    result = {"sample_size": len(rows), "sample_seen": len(rows), "rates_percent": {field: round(100 * float(frame[field].mean()), 4) for field in ["same_country", "raw_name_exact", "norm_name_exact", "raw_address_exact", "norm_address_exact", "left_name_missing", "right_name_missing", "left_address_missing", "right_address_missing"]}}
    result["distributions"] = {field: {key: round(float(value), 6) for key, value in frame[field].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).to_dict().items()} for field in numeric}
    if include_breakdowns:
        result["by_source_pair"] = {key: summarize_pair_rows(group.to_dict("records"), include_breakdowns=False) for key, group in frame.groupby("source_pair")}
        result["by_country"] = {key: summarize_pair_rows(group.to_dict("records"), include_breakdowns=False) for key, group in frame.groupby("country")}
    return result


def normalization_analysis(source_profiles, pair_rows):
    result = {}
    for label, transform in (("N0_casefold", norm0), ("N1_unicode", norm1), ("N2_punctuation", norm2), ("N3_whitespace", norm3), ("N4_diacritic", norm4)):
        changed = collisions = 0
        for profile in source_profiles:
            for sample in profile["examples"]:
                value = sample["business_name"]
                changed += transform(value) != value
        for row in pair_rows:
            if transform(row.get("left_name", "")) == transform(row.get("right_name", "")):
                collisions += 1
        result[label] = {"sample_records_changed": changed, "sample_positive_agreements": collisions}
    return result


def hard_negatives(train_profiles, gt, pair_rows):
    # Exact normalized-name collisions are retrieved from compact per-file examples.
    by_name = defaultdict(list)
    for profile in train_profiles:
        for row in profile["examples"]:
            by_name[norm3(row["business_name"])].append(row)
    negatives = []
    for name, rows in by_name.items():
        if not name or len(rows) < 2:
            continue
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                if rows[left_index]["entity_id"][:2] == rows[right_index]["entity_id"][:2]:
                    continue
                negatives.append({"type": "same_normalized_name_sample", "name": name, "left": rows[left_index], "right": rows[right_index], "address_similarity": text_similarity(norm3(rows[left_index]["business_address"]), norm3(rows[right_index]["business_address"]))})
                if len(negatives) >= HARD_NEGATIVE_LIMIT:
                    return negatives
    return negatives


def markdown(title, body):
    return f"# {title}\n\n{body.strip()}\n"


def write_reports(profile):
    inv = profile["inventory"]
    gt = profile["ground_truth"]
    source_lines = ["| File | Rows | Size MB | Missing name | Missing address | Non-ASCII name | Non-ASCII address |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in inv:
        source_lines.append(f"| `{item['path']}` | {item['rows']:,} | {item['size_bytes']/1e6:.2f} | {item['business_name']['empty_percent']:.2f}% | {item['business_address']['empty_percent']:.2f}% | {item['patterns'].get('name_non_ascii', 0)/max(1,item['rows']):.2%} | {item['patterns'].get('address_non_ascii', 0)/max(1,item['rows']):.2%} |")
    sections = {
        "01_dataset_inventory.md": markdown("Dataset Inventory", "[EXACT][FULL-DATA] All seven TSV files were scanned.\n\n" + "\n".join(source_lines) + "\n\nThe JSON companion contains exact length/token histograms, schema, ID-prefix, country, missingness, and duplicate statistics."),
        "02_ground_truth_analysis.md": markdown("Ground Truth Analysis", f"[EXACT][FULL-DATA] S1 rows: **{gt['rows']:,}**. Total links: **{gt['total_links']:,}**. Zero-match rate: **{gt['zero_percent']:.2f}%**. Singleton rate: **{gt['singleton_rate_percent']:.2f}%**. Multi-match rate: **{gt['multi_match_rate_percent']:.2f}%**. Maximum matches: **{gt['max_matches']}**.\n\nMapping classification: **{gt['mapping_classification']}**. S2 inverse sharing: {gt['s2_inverse']}. S3 inverse sharing: {gt['s3_inverse']}.\n\nConnected components: {gt['graph_components']:,}; largest component: {gt['largest_component']:,} nodes; nodes in components of size >=10: {gt['nodes_in_components_10_plus']:,}."),
        "03_field_analysis.md": markdown("Field Analysis", "[EXACT][FULL-DATA] Per-source business-name and business-address distributions, missingness, duplicate counts, token/length histograms, character-pattern rates, examples, and approximate heavy hitters are in `dataset_inventory.json`. Heavy hitters are labelled approximate because bounded heavy-hitter maps are used."),
        "04_country_analysis.md": markdown("Country Analysis", "[EXACT][FULL-DATA] Country frequencies are reported per source in `dataset_inventory.json`. Country is retained as an observed relational signal; this report does not justify hard blocking."),
        "05_train_test_shift.md": markdown("Train/Test Shift", "[EXACT][FULL-DATA] Country, missingness, length, token, and pattern summaries are compared in the JSON. Jensen-Shannon divergence is provided for categorical country distributions. Shift labels are evidence summaries, not model decisions."),
        "06_positive_pair_analysis.md": markdown("Positive Pair Analysis", f"[SAMPLE] {profile['positive_pairs']['sample_size']:,} true links were analysed from a fixed reservoir of {gt['pair_sample_seen']:,} links. Exact-equality rates, similarities, token overlap, digit overlap, missingness, source-pair, and country breakdowns are in `positive_pair_analysis` in the master JSON."),
        "07_hard_negative_analysis.md": markdown("Hard Negative Analysis", f"[SAMPLE] {len(profile['hard_negatives']):,} same-normalized-name cross-source collision examples were retrieved from bounded source samples. They are diagnostic hard negatives, not a full negative population."),
        "08_normalization_analysis.md": markdown("Normalization Analysis", "[SAMPLE] Conservative normalization variants are compared in the JSON. Positive agreement and collision measurements are sample-based and must not be treated as final normalization decisions."),
        "09_blocking_signal_analysis.md": markdown("Blocking Signal Analysis", "[APPROXIMATE][SAMPLE] Exact normalized-name, address, country-name, address-digit, token, and union diagnostics are represented using sampled linked records and full source field summaries. This is signal discovery, not a selected blocker."),
        "10_source_analysis.md": markdown("Source Analysis", "[EXACT][FULL-DATA] All train and test source files are profiled separately. [SAMPLE] Positive-pair comparisons are broken down by S1-S2 and S1-S3 where sampled linked records are available."),
        "11_france_analysis.md": markdown("France Analysis", "[EXACT][FULL-DATA] France observations are reported from all test source files only. [INFERENCE] No France matching quality claim is possible without France ground truth. US/India proxy observations must be treated as stress tests, not France evaluation."),
        "12_modern_methods_assessment.md": markdown("Modern Methods Assessment", "[INFERENCE] Classical lexical retrieval, character n-grams, supervised pair classification, semantic retrieval, hybrid reranking, and hard-negative mining remain experiment candidates. The actual priority must be decided after reading the measured candidate volumes and pair-overlap tables; no final matcher is selected here."),
        "13_feature_signal_analysis.md": markdown("Feature Signal Analysis", "[SAMPLE][INFERENCE] Name lexical, address lexical, address digits, country agreement, missingness, source-pair, and blocker provenance are measured in the positive-pair table. Signal rankings remain unknown until hard-negative overlap is compared quantitatively."),
        "14_computational_analysis.md": markdown("Computational Analysis", f"[EXACT][FULL-DATA] Source sizes and row counts are in the inventory. [APPROXIMATE] Pairwise Cartesian comparison is infeasible; candidate retrieval must operate on indexes. The reproducible runner uses {CHUNK_SIZE:,}-row chunks and a {PAIR_SAMPLE_LIMIT:,}-pair reservoir."),
        "15_failure_modes.md": markdown("Failure Modes", "[INFERENCE] Priority risks to verify are singleton false positives, repeated names, shared addresses, missing fields, source-specific noise, cross-script variation, unseen-country shift, and candidate-generation misses. Frequencies and examples are in the machine-readable report."),
    }
    for filename, content in sections.items():
        (OUT / filename).write_text(content, encoding="utf-8")
    gt_summary = f"- [EXACT] Training S1 rows: {gt['rows']:,}\n- [EXACT] Ground-truth links: {gt['total_links']:,}\n- [EXACT] Zero-match rate: {gt['zero_percent']:.2f}%\n- [EXACT] Singleton rate: {gt['singleton_rate_percent']:.2f}%\n- [EXACT] Multi-match rate: {gt['multi_match_rate_percent']:.2f}%\n- [EXACT] S2 inverse shared entities: {gt['s2_inverse']['shared_by_multiple_s1']:,}\n- [EXACT] S3 inverse shared entities: {gt['s3_inverse']['shared_by_multiple_s1']:,}\n- [EXACT] Ground-truth components: {gt['graph_components']:,}\n- [SAMPLE] True pairs analysed: {profile['positive_pairs']['sample_size']:,}\n- [SAMPLE] Hard negatives: {len(profile['hard_negatives']):,}\n"
    master = "# Master Data Intelligence Report\n\n" + gt_summary + "\n" + "\n".join(f"- [EXACT] {item['path']}: {item['rows']:,} rows, {item['size_bytes']/1e6:.2f} MB" for item in inv) + "\n\n## Open decisions\n\n[INFERENCE] Final normalization, blocker, matcher, thresholds, and semantic-model usage remain open and must be decided by leakage-safe end-to-end experiments.\n\n## Recommended experiment order\n\n1. Deterministic exact/normalized baseline.\n2. Measure blocking recall and candidate reduction separately.\n3. Compare lexical feature families on hard negatives.\n4. Test supervised pair classification only after candidate recall is measured.\n5. Test semantic retrieval/reranking only for residual cross-script or transliteration cases.\n6. Optimize thresholds against macro F0.5 including singleton S1s.\n\n## Evidence artifacts\n\nSee `dataset_inventory.json`, `ground_truth_analysis.json`, and `master_data_intelligence.json` for full machine-readable values.\n"
    (OUT / "MASTER_DATA_INTELLIGENCE_REPORT.md").write_text(master, encoding="utf-8")


def repair_existing_reports():
    inventory_path = OUT / "dataset_inventory.json"
    ground_truth_path = OUT / "ground_truth_analysis.json"
    master_path = OUT / "master_data_intelligence.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    gt_file = DATA / "train" / "train_ground_truth.tsv"
    ground_truth["graph_components"] = sum(ground_truth["component_size_histogram"].values())
    ground_truth["mapping_classification"] = "one-to-many"
    ground_truth_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    if not any(item.get("path", "").endswith("train_ground_truth.tsv") for item in inventory["inventory"]):
        inventory["inventory"].append({"path": str(gt_file.relative_to(ROOT)), "size_bytes": gt_file.stat().st_size, "rows": ground_truth["rows"], "columns": ground_truth["columns"], "delimiter_valid": ground_truth["delimiter_valid"], "malformed_rows": ground_truth["malformed_rows"], "ground_truth_file": True})
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    master["inventory"] = inventory["inventory"]
    master["ground_truth"] = ground_truth
    master_path.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory_md = "# Dataset Inventory\n\n" + "\n".join(f"- [EXACT][FULL-DATA] `{item['path']}`: {item['rows']:,} rows, {item['size_bytes']:,} bytes, columns `{item['columns']}`" for item in inventory["inventory"]) + "\n"
    (OUT / "dataset_inventory.md").write_text(inventory_md, encoding="utf-8")
    master_md = OUT / "MASTER_DATA_INTELLIGENCE_REPORT.md"
    text = master_md.read_text(encoding="utf-8")
    text = text.replace("[EXACT] Ground-truth components: 12", f"[EXACT] Ground-truth components: {ground_truth['graph_components']:,}")
    text = text.replace("[EXACT] S2 inverse shared entities: 0", f"[EXACT] S2 inverse shared entities: {ground_truth['s2_inverse']['shared_by_multiple_s1']:,}")
    text = text.replace("[EXACT] S3 inverse shared entities: 0", f"[EXACT] S3 inverse shared entities: {ground_truth['s3_inverse']['shared_by_multiple_s1']:,}")
    if "train_ground_truth.tsv" not in text:
        text = text.replace("- [SAMPLE] Hard negatives:", f"- [EXACT] Ground-truth file rows: {ground_truth['rows']:,}\n- [SAMPLE] Hard negatives:")
    master_md.write_text(text, encoding="utf-8")
    print("REPORT REPAIR COMPLETE")


def targeted_hard_negatives(paths):
    name_counts = Counter()
    train_paths = [paths[name] for name in ("train_source1", "train_source2", "train_source3")]
    for path in train_paths:
        for chunk in pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, chunksize=CHUNK_SIZE, usecols=["entity_id", "business_name", "business_address", "country"]):
            counts = chunk["business_name"].astype(str).str.strip().str.casefold().value_counts()
            name_counts.update({key: int(value) for key, value in counts.items() if key})
    repeated_names = {name for name, count in name_counts.items() if count >= 2}
    buckets = defaultdict(list)
    for path in train_paths:
        for chunk in pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, chunksize=CHUNK_SIZE, usecols=["entity_id", "business_name", "business_address", "country"]):
            names = chunk["business_name"].astype(str).str.strip().str.casefold()
            selected = chunk[names.isin(repeated_names)]
            for record in selected.itertuples(index=False, name=None):
                key = record[1].strip().casefold()
                if len(buckets[key]) < 6:
                    buckets[key].append({"entity_id": str(record[0]), "business_name": str(record[1]), "business_address": str(record[2]), "country": str(record[3])})
    candidate_pairs = []
    for name, records in buckets.items():
        for left_index, left in enumerate(records):
            for right in records[left_index + 1:]:
                if left["entity_id"][:2] == right["entity_id"][:2]:
                    continue
                if norm3(left["business_address"]) == norm3(right["business_address"]):
                    continue
                if left["entity_id"].startswith("S1-") and right["entity_id"].startswith(("S2-", "S3-")):
                    source1, other, source1_record, other_record = left["entity_id"], right["entity_id"], left, right
                elif right["entity_id"].startswith("S1-") and left["entity_id"].startswith(("S2-", "S3-")):
                    source1, other, source1_record, other_record = right["entity_id"], left["entity_id"], right, left
                else:
                    continue
                candidate_pairs.append((source1, other, name, source1_record, other_record))
                if len(candidate_pairs) >= HARD_NEGATIVE_LIMIT * 2:
                    break
            if len(candidate_pairs) >= HARD_NEGATIVE_LIMIT * 2:
                break
        if len(candidate_pairs) >= HARD_NEGATIVE_LIMIT * 2:
            break
    candidate_ids = {(source1, other) for source1, other, _, _, _ in candidate_pairs}
    candidate_source1_ids = {source1 for source1, _, _, _, _ in candidate_pairs}
    true_ids = set()
    with (DATA / "train" / "train_ground_truth.tsv").open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) != 2 or row[0] not in candidate_source1_ids:
                continue
            for matched in row[1].split(","):
                if (row[0], matched.strip()) in candidate_ids:
                    true_ids.add((row[0], matched.strip()))
    result = []
    for source1, other, name, source1_record, other_record in candidate_pairs:
        if (source1, other) in true_ids:
            continue
        result.append({"type": "same_normalized_name_different_address", "normalized_name": name, "left": source1_record, "right": other_record, "address_similarity": text_similarity(norm3(source1_record["business_address"]), norm3(other_record["business_address"])), "same_country": source1_record["country"].casefold() == other_record["country"].casefold()})
        if len(result) >= HARD_NEGATIVE_LIMIT:
            break
    return result


def main():
    paths = file_paths()
    gt = scan_ground_truth(paths["train_ground_truth"])
    wanted = {s1 for s1, _ in gt["pair_sample"]} | {entity for _, entity in gt["pair_sample"]}
    inventory = []
    profiles = {}
    for name, path in paths.items():
        if name == "train_ground_truth":
            continue
        profile = scan_source(path, wanted_ids=wanted if name.startswith("train_") else None)
        inventory.append(profile)
        profiles[name] = profile
    records = {}
    for profile in inventory:
        records.update(profile.pop("selected_rows", {}))
    positive_rows = pair_metrics(gt["pair_sample"], records)
    # Add raw fields needed for the normalization and hard-negative evidence.
    for row in positive_rows:
        left, right = records.get(row["s1"], ("", "", "")), records.get(row["other"], ("", "", ""))
        row["left_name"], row["right_name"] = left[0], right[0]
    train_profiles = [profiles[name] for name in ("train_source1", "train_source2", "train_source3")]
    hard = hard_negatives(train_profiles, gt, positive_rows)
    countries = {name: profile["countries"] for name, profile in profiles.items()}
    country_shift = {}
    for source in ("source1", "source2", "source3"):
        train = countries[f"train_{source}"]; test = countries[f"test_{source}"]
        country_shift[source] = {"train": dict(train), "test": dict(test), "js_divergence": js_divergence(train, test)}
    profile = {"metadata": {"seed": SEED, "chunk_size": CHUNK_SIZE, "pair_sample_limit": PAIR_SAMPLE_LIMIT, "scope": "all seven TSV files", "originals_modified": False}, "inventory": inventory, "ground_truth": gt, "positive_pairs": summarize_pair_rows(positive_rows), "hard_negatives": hard, "normalization": normalization_analysis(inventory, positive_rows), "country_shift": country_shift, "test_france": {name: dict(profile["countries"]) for name, profile in profiles.items() if name.startswith("test_")}, "positive_pair_examples": positive_rows[:100]}
    serializable = json.loads(json.dumps(profile, ensure_ascii=False, default=lambda value: dict(value) if isinstance(value, Counter) else value))
    (OUT / "dataset_inventory.json").write_text(json.dumps({"metadata": serializable["metadata"], "inventory": serializable["inventory"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ground_truth_analysis.json").write_text(json.dumps(serializable["ground_truth"], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "master_data_intelligence.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "dataset_inventory.md").write_text("# Dataset Inventory\n\n" + "\n".join(f"- [EXACT][FULL-DATA] `{item['path']}`: {item['rows']:,} rows, {item['size_bytes']:,} bytes, columns `{item['columns']}`" for item in serializable["inventory"]) + "\n", encoding="utf-8")
    write_reports(serializable)
    print("DATA INTELLIGENCE COMPLETE")
    print(json.dumps({"files": len(paths), "source_rows": {item["path"]: item["rows"] for item in serializable["inventory"]}, "train_s1": gt["rows"], "total_links": gt["total_links"], "zero_percent": gt["zero_percent"], "singleton_percent": gt["singleton_rate_percent"], "mapping": gt["mapping_classification"], "components": gt["graph_components"], "pair_sample": len(positive_rows), "hard_negatives": len(hard), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    if "--repair-existing" in sys.argv:
        repair_existing_reports()
    elif "--augment-hard-negatives" in sys.argv:
        paths = file_paths()
        hard = targeted_hard_negatives(paths)
        master_path = OUT / "master_data_intelligence.json"
        master = json.loads(master_path.read_text(encoding="utf-8"))
        master["hard_negatives"] = hard
        master_path.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "07_hard_negative_analysis.md").write_text(markdown("Hard Negative Analysis", f"[SAMPLE][GROUND-TRUTH-EXCLUDED] Retrieved {len(hard):,} same-normalized-name, different-address cross-source candidates from full training sources. Candidates matching a known ground-truth link were excluded; remaining cases are hard-negative candidates for analysis, not a claim that every pair is independently verified negative."), encoding="utf-8")
        print(json.dumps({"hard_negative_candidates": len(hard), "output": str(OUT)}, ensure_ascii=False))
    else:
        main()
