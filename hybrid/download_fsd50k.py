"""Download a small, reproducible FSD50K subset from its official Zenodo release.

Only ZIP central directories and selected members are transferred: never the
24.7 GB audio archive. Official train/val assignments are retained, eval becomes
test, and clips from uploaders spanning dev splits are filtered rather than
reassigned. Labels describe entire clips, not exact event timestamps.

Run: python -m hybrid.download_fsd50k --output data/fsd50k
Use --metadata-only to inspect selection/provenance without acquiring audio.
The optional --allow-restricted-licenses includes NC and Sampling+ material;
the default permits only the original CC0 and CC-BY clips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RECORD_URL = "https://zenodo.org/records/4060432"
DATASET_DOI = "10.5281/zenodo.4060432"
FILES = {
    "FSD50K.ground_truth.zip": (334701, "ca27382c195e37d2269c4c866dd73485"),
    "FSD50K.metadata.zip": (6700838, "b9ea0c829a411c1d42adb9da539ed237"),
    "FSD50K.doc.zip": (6984, "3516162b82dc2945d3e7feba0904e800"),
    "FSD50K.dev_audio.z01": (3221225472, "faa7cf4cc076fc34a44a479a5ed862a3"),
    "FSD50K.dev_audio.z02": (3221225472, "8f9b66153e68571164fb1315d00bc7bc"),
    "FSD50K.dev_audio.z03": (3221225472, "1196ef47d267a993d30fa98af54b7159"),
    "FSD50K.dev_audio.z04": (3221225472, "d088ac4e11ba53daf9f7574c11cccac9"),
    "FSD50K.dev_audio.z05": (3221225472, "81356521aa159accd3c35de22da28c7f"),
    "FSD50K.dev_audio.zip": (2306663327, "c480d119b8f7a7e32fdb58f3ea4d6c5a"),
    "FSD50K.eval_audio.z01": (3221225472, "3090670eaeecc013ca1ff84fe4442aeb"),
    "FSD50K.eval_audio.zip": (3037675767, "6fa47636c3a3ad5c7dfeba99f2637982"),
}
POSITIVE = {"Clapping": "clap", "Finger_snapping": "finger_snap"}
HARD_NEGATIVES = (
    "Knock", "Tap", "Tick", "Click", "Typing", "Keys_jangling", "Door",
    "Dishes_and_pots_and_pans", "Glass", "Cough", "Sneeze", "Footsteps",
)
SPLITS = ("train", "val", "test")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


class HttpReader:
    """Bounded HTTP reads; a server ignoring Range cannot trigger a full download."""

    def __init__(self, requests_per_second: float = 1.6):
        self.interval = 1 / requests_per_second
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.transferred = 0

    def read(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        size = FILES[name][0]
        if length is None:
            length = size
        if not 0 <= start < size or not 0 < length <= size - start:
            raise ValueError(f"range outside {name}: {start}+{length}")
        if length > 16 * 1024 * 1024:
            raise ValueError(f"refusing oversized transfer for {name}; only selective ranges are allowed")
        headers = {"User-Agent": "clap-lights-fsd50k-subset/1.0", "Accept-Encoding": "identity"}
        ranged = start != 0 or length != size
        if ranged:
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        request = urllib.request.Request(f"{RECORD_URL}/files/{name}?download=1", headers=headers)
        for attempt in range(6):
            with self.lock:
                delay = max(0, self.next_request - time.monotonic())
                self.next_request = max(self.next_request, time.monotonic()) + self.interval
            if delay:
                time.sleep(delay)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{start + length - 1}/{size}"
                    if ranged and (response.status != 206 or response.headers.get("Content-Range") != expected):
                        raise ValueError(f"server did not honor bounded Range for {name}")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != length:
                        raise ValueError(f"unexpected response length for {name}")
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise IOError(f"short or oversized HTTP response for {name}")
                with self.lock:
                    self.transferred += len(data)
                return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
                    raise
                delay = min(60, float(exc.headers.get("Retry-After", 2 ** attempt)))
                time.sleep(max(1, delay))
            except (TimeoutError, ConnectionError, urllib.error.URLError, IOError):
                if attempt == 5:
                    raise
                time.sleep(min(16, 2 ** attempt))
        raise RuntimeError("HTTP retry loop exhausted")


def cached_zip(root: Path, name: str, http: HttpReader) -> zipfile.ZipFile:
    path = root / "metadata" / name
    data = path.read_bytes() if path.exists() else b""
    if len(data) != FILES[name][0] or hashlib.md5(data).hexdigest() != FILES[name][1]:
        data = http.read(name)
        if hashlib.md5(data).hexdigest() != FILES[name][1]:
            raise ValueError(f"official archive checksum mismatch: {name}")
        atomic_write(path, data)
    return zipfile.ZipFile(io.BytesIO(data))


def _zip64_values(extra: bytes, size: int, compressed: int, offset: int, disk: int):
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        data = extra[cursor:cursor + length]
        cursor += length
        if kind != 1:
            continue
        position = 0
        result = []
        for value, sentinel, fmt in ((size, 0xffffffff, "Q"), (compressed, 0xffffffff, "Q"),
                                     (offset, 0xffffffff, "Q"), (disk, 0xffff, "I")):
            if value == sentinel:
                value = struct.unpack_from("<" + fmt, data, position)[0]
                position += struct.calcsize(fmt)
            result.append(value)
        return tuple(result)
    if 0xffffffff in (size, compressed, offset) or disk == 0xffff:
        raise ValueError("missing ZIP64 member fields")
    return size, compressed, offset, disk


class SplitZip:
    """Read central-directory/member byte ranges across official split ZIP disks."""

    def __init__(self, split: str, root: Path, http: HttpReader):
        self.http = http
        disks = 6 if split == "dev" else 2
        self.names = [f"FSD50K.{split}_audio.z{i:02d}" for i in range(1, disks)]
        self.names.append(f"FSD50K.{split}_audio.zip")
        self.sizes = [FILES[name][0] for name in self.names]
        cache = root / "metadata" / f"{split}.central-directory.bin"
        if cache.exists():
            directory = cache.read_bytes()
        else:
            tail_size = min(65557, self.sizes[-1])
            tail = http.read(self.names[-1], self.sizes[-1] - tail_size, tail_size)
            location = tail.rfind(b"PK\x05\x06")
            if location < 0 or len(tail) - location < 22:
                raise ValueError("ZIP end-of-central-directory not found")
            _, disk, cd_disk, _, entries, cd_size, cd_offset, comment_length = struct.unpack_from(
                "<4s4H2IH", tail, location)
            if location + 22 + comment_length != len(tail) or disk != disks - 1:
                raise ValueError("unexpected split ZIP layout")
            if entries == 0xffff or cd_size == 0xffffffff or cd_offset == 0xffffffff:
                if location < 20 or tail[location - 20:location - 16] != b"PK\x06\x07":
                    raise ValueError("missing ZIP64 locator")
                zip_disk, zip_offset, disk_count = struct.unpack_from("<IQI", tail, location - 16)
                if disk_count != disks:
                    raise ValueError("unexpected ZIP64 disk count")
                record = self.read(zip_disk, zip_offset, 56)
                if record[:4] != b"PK\x06\x06":
                    raise ValueError("missing ZIP64 directory record")
                cd_disk = struct.unpack_from("<I", record, 20)[0]
                entries, cd_size, cd_offset = struct.unpack_from("<QQQ", record, 32)
            if cd_size > 16 * 1024 * 1024:
                raise ValueError("unexpectedly large ZIP central directory")
            directory = self.read(cd_disk, cd_offset, cd_size)
            atomic_write(cache, directory)
        self.members = {}
        cursor = 0
        while cursor < len(directory):
            if directory[cursor:cursor + 4] != b"PK\x01\x02" or cursor + 46 > len(directory):
                raise ValueError("invalid ZIP central directory")
            flags, method = struct.unpack_from("<HH", directory, cursor + 8)
            crc, compressed, size = struct.unpack_from("<III", directory, cursor + 16)
            name_length, extra_length, comment_length, disk = struct.unpack_from("<4H", directory, cursor + 28)
            offset = struct.unpack_from("<I", directory, cursor + 42)[0]
            name = directory[cursor + 46:cursor + 46 + name_length].decode("utf-8" if flags & 0x800 else "cp437")
            extra = directory[cursor + 46 + name_length:cursor + 46 + name_length + extra_length]
            size, compressed, offset, disk = _zip64_values(extra, size, compressed, offset, disk)
            self.members[name] = dict(crc=crc, compressed=compressed, size=size, disk=disk,
                                      offset=offset, flags=flags, method=method)
            cursor += 46 + name_length + extra_length + comment_length

    def read(self, disk: int, offset: int, length: int) -> bytes:
        result = []
        while length:
            if disk >= len(self.names) or offset > self.sizes[disk]:
                raise ValueError("ZIP read exceeds split archive")
            if offset == self.sizes[disk]:
                disk, offset = disk + 1, 0
                continue
            count = min(length, self.sizes[disk] - offset)
            result.append(self.http.read(self.names[disk], offset, count))
            length -= count
            disk, offset = disk + 1, 0
        return b"".join(result)

    def extract(self, name: str, destination: Path) -> dict:
        member = self.members[name]
        if destination.exists():
            data = destination.read_bytes()
            if len(data) == member["size"] and zlib.crc32(data) & 0xffffffff == member["crc"]:
                return member
        if member["flags"] & 1 or member["method"] not in (0, 8):
            raise ValueError(f"unsupported ZIP compression/encryption for {name}")
        if member["size"] > 8 * 1024 * 1024 or member["compressed"] > 8 * 1024 * 1024:
            raise ValueError(f"unexpected audio size for {name}")
        # One request normally obtains the small local header and all compressed
        # data; clipping the spare header allowance avoids reading past the ZIP.
        disk, offset = member["disk"], member["offset"]
        available = sum(self.sizes[disk:]) - offset
        raw = self.read(disk, offset, min(available, member["compressed"] + 512))
        if raw[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid local ZIP header for {name}")
        name_length, extra_length = struct.unpack_from("<HH", raw, 26)
        payload_start = 30 + name_length + extra_length
        payload_end = payload_start + member["compressed"]
        if payload_end > len(raw):
            raw = self.read(disk, offset, payload_end)
        compressed = raw[payload_start:payload_end]
        data = zlib.decompress(compressed, -15) if member["method"] == 8 else compressed
        if len(data) != member["size"] or zlib.crc32(data) & 0xffffffff != member["crc"]:
            raise ValueError(f"audio ZIP CRC/size mismatch for {name}")
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 44100):
                raise ValueError(f"unexpected official audio format for {name}")
        atomic_write(destination, data)
        return member


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def choose_diverse(records: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin uploader/category groups before selecting repeated sources."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["uploader"].casefold(), record["category"])].append(record)
    for group in groups.values():
        group.sort(key=lambda record: stable_key(record["source_id"], seed))
    keys = sorted(groups, key=lambda key: stable_key(str(key), seed))
    result = []
    while len(result) < count and keys:
        active = []
        for key in keys:
            if len(result) == count:
                break
            result.append(groups[key].pop(0))
            if groups[key]:
                active.append(key)
        keys = active
    return result


def select_records(ground_truth: zipfile.ZipFile, metadata: zipfile.ZipFile,
                   positives: int, negatives: int, seed: int, restricted: bool) -> tuple[list[dict], dict]:
    candidates = []
    excluded = Counter()
    for original_split in ("dev", "eval"):
        info = json.loads(metadata.read(f"FSD50K.metadata/{original_split}_clips_info_FSD50K.json"))
        text = ground_truth.read(f"FSD50K.ground_truth/{original_split}.csv").decode()
        for row in csv.DictReader(io.StringIO(text)):
            labels = row["labels"].split(",")
            targets = [name for name in POSITIVE if name in labels]
            if len(targets) > 1:
                excluded["both_positive_labels"] += 1
                continue
            category = targets[0] if targets else next((name for name in HARD_NEGATIVES if name in labels), None)
            if category is None:
                continue
            clip = info[row["fname"]]
            license_url = clip["license"]
            permissive = "/publicdomain/zero/" in license_url or "/licenses/by/" in license_url
            if not permissive and not restricted:
                excluded["restricted_license"] += 1
                continue
            uploader = str(clip["uploader"]).strip()
            if not uploader:
                excluded["missing_uploader"] += 1
                continue
            candidates.append({
                "filename": f"audio/{row['fname']}.wav",
                "label": POSITIVE.get(category, "noise"),
                "source_id": f"fsd50k:{row['fname']}",
                "uploader": uploader,
                "split": row.get("split", "test"),
                "official_split": row.get("split", "eval"),
                "archive_split": original_split,
                "category": category,
                "ground_truth_labels": labels,
                "label_quality": "weak_clip_positive" if targets else "weak_clip_negative",
                "license": license_url,
                "title": clip["title"],
                "source_url": f"https://freesound.org/people/{urllib.parse.quote(uploader, safe='')}/sounds/{row['fname']}/",
                "dataset_url": RECORD_URL,
                "dataset_doi": DATASET_DOI,
            })
    uploader_votes = defaultdict(Counter)
    eval_uploaders = set()
    for record in candidates:
        user = record["uploader"].casefold()
        if record["split"] == "test":
            eval_uploaders.add(user)
        else:
            # Preserve the scarcer snap validation sources when a person has
            # uploaded other sounds appearing in the official training split.
            weight = 100 if record["label"] == "finger_snap" else 10 if record["label"] == "clap" else 1
            uploader_votes[user][record["split"]] += weight
    uploader_split = {user: "val" if votes["val"] >= votes["train"] else "train"
                      for user, votes in uploader_votes.items()}
    clean = []
    for record in candidates:
        user = record["uploader"].casefold()
        if record["split"] != "test" and (user in eval_uploaders or uploader_split[user] != record["split"]):
            excluded["uploader_split_conflict"] += 1
            continue
        clean.append(record)
    selected = []
    available = Counter((record["label"], record["split"]) for record in clean)
    for label, total in (("clap", positives), ("finger_snap", positives), ("noise", negatives)):
        quotas = {"train": round(total * .7), "val": round(total * .15)}
        quotas["test"] = total - sum(quotas.values())
        chosen = []
        for split in SPLITS:
            group = [record for record in clean if record["label"] == label and record["split"] == split]
            chosen.extend(choose_diverse(group, quotas[split], seed))
        chosen_ids = {record["source_id"] for record in chosen}
        remainder = [record for record in clean if record["label"] == label and record["source_id"] not in chosen_ids]
        chosen.extend(choose_diverse(remainder, max(0, total - len(chosen)), seed))
        selected.extend(chosen)
    selected.sort(key=lambda record: (record["split"], record["label"], record["source_id"]))
    verify_isolation(selected)
    summary = {
        "counts": {f"{label}/{split}": sum(record["label"] == label and record["split"] == split for record in selected)
                   for label in ("clap", "finger_snap", "noise") for split in SPLITS},
        "available_after_filtering": {f"{label}/{split}": count for (label, split), count in sorted(available.items())},
        "excluded": dict(excluded),
        "uploaders": len({record["uploader"].casefold() for record in selected}),
    }
    return selected, summary


def verify_isolation(records: list[dict]) -> None:
    uploaders, sources, digests = {}, set(), {}
    for record in records:
        uploader = record["uploader"].casefold()
        if record["source_id"] in sources:
            raise ValueError("source ID selected more than once")
        sources.add(record["source_id"])
        if uploader in uploaders and uploaders[uploader] != record["split"]:
            raise ValueError(f"uploader crosses splits: {record['uploader']}")
        uploaders[uploader] = record["split"]
        digest = record.get("sha256")
        if digest:
            identity = (record["split"], record["label"])
            if digest in digests and digests[digest] != identity:
                raise ValueError("byte-identical audio crosses splits or has conflicting class labels")
            digests[digest] = identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fsd50k"))
    parser.add_argument("--positive-count", type=int, default=100, help="requested clips per positive class")
    parser.add_argument("--negative-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--allow-restricted-licenses", action="store_true")
    args = parser.parse_args()
    if args.positive_count < 3 or args.negative_count < 3 or not 1 <= args.workers <= 8:
        parser.error("counts must be >=3 and workers must be between 1 and 8")
    root = args.output.resolve()
    http = HttpReader()
    ground_truth = cached_zip(root, "FSD50K.ground_truth.zip", http)
    metadata = cached_zip(root, "FSD50K.metadata.zip", http)
    documentation = cached_zip(root, "FSD50K.doc.zip", http)
    for name in ("README.md", "LICENSE-DATASET"):
        atomic_write(root / "metadata" / name, documentation.read(f"FSD50K.doc/{name}"))
    selected, summary = select_records(ground_truth, metadata, args.positive_count, args.negative_count,
                                       args.seed, args.allow_restricted_licenses)
    provenance = {
        "dataset": "FSD50K v1.0", "doi": DATASET_DOI, "source": RECORD_URL,
        "citation": "Fonseca et al. (2022), FSD50K: an open dataset of human-labeled sound events, IEEE/ACM TASLP 30:829-852.",
        "dataset_license": "CC-BY (see metadata/LICENSE-DATASET); clip licenses are listed individually in manifest.json",
        "seed": args.seed, "requested_positive_count": args.positive_count,
        "requested_negative_count": args.negative_count,
        "license_filter": "all original licenses" if args.allow_restricted_licenses else "CC0 and CC-BY only",
        "split_policy": "Keep official train/val assignments; eval=test. Filter uploader conflicts using weighted positive-class majority, val wins ties. No uploader/source crosses splits.",
        "sampling_policy": "Deterministic uploader/category round-robin with 70/15/15 split quotas; fill unavailable quotas from remaining eligible clips.",
        "label_limitations": "FSD50K labels are weak clip labels. Positive timestamps need annotation. Dev negative labels may be incomplete. Applause/clapping is broader than isolated intentional claps. Web audio does not validate the ESP32 microphone domain.",
        "official_archive_checksums": {name: {"bytes": size, "md5": md5} for name, (size, md5) in FILES.items()},
        "verification": "Small metadata archives checked against official MD5. Selective audio verified by official ZIP member CRC32 and PCM format, then SHA256 recorded. Full multi-GB archive checksums cannot be verified from partial download.",
        **summary,
    }
    write_json(root / "metadata" / "selection.json", selected)
    write_json(root / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    if args.metadata_only:
        print(f"Metadata and planned selection saved in {root}; no audio downloaded.")
        return
    archives = {split: SplitZip(split, root, http) for split in ("dev", "eval")}
    completed, failures = [], []

    def acquire(record: dict) -> dict:
        filename = record["filename"].split("/")[-1]
        archive_split = record["archive_split"]
        destination = root / record["filename"]
        member = archives[archive_split].extract(f"FSD50K.{archive_split}_audio/{filename}", destination)
        data = destination.read_bytes()
        with wave.open(io.BytesIO(data), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return {**record, "sha256": hashlib.sha256(data).hexdigest(),
                "zip_crc32": f"{member['crc']:08x}", "duration_seconds": duration,
                "sample_rate": 44100, "channels": 1, "sample_width_bytes": 2}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(acquire, record): record for record in selected}
        for index, future in enumerate(as_completed(jobs), 1):
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({"source_id": jobs[future]["source_id"], "error": str(exc)})
            if index == 1 or index % 10 == 0 or index == len(jobs):
                print(f"Audio {index}/{len(jobs)}: {len(completed)} verified, {len(failures)} failed; transferred {http.transferred / 1e6:.1f} MB", flush=True)
    completed.sort(key=lambda record: (record["split"], record["label"], record["source_id"]))
    verify_isolation(completed)
    write_json(root / "manifest.json", completed)
    attribution = ["FSD50K subset attribution", f"Dataset: {RECORD_URL}", provenance["citation"],
                   "Original audio remains under each uploader's indicated license.", ""]
    attribution.extend(f"{record['filename']} | {record['title']} | {record['uploader']} | {record['license']} | {record['source_url']}"
                       for record in completed)
    atomic_write(root / "ATTRIBUTION.txt", ("\n".join(attribution) + "\n").encode())
    provenance.update(downloaded_clips=len(completed), failed_clips=failures, transferred_bytes=http.transferred)
    write_json(root / "provenance.json", provenance)
    if failures:
        raise SystemExit(f"{len(failures)} clips failed. Verified partial manifest saved; rerun to resume.")
    if any(not any(record["label"] == label and record["split"] == split for record in completed)
           for label in ("clap", "finger_snap", "noise") for split in SPLITS):
        raise SystemExit("Dataset missing a class/split. Inspect provenance before training.")
    print(f"Ready: {len(completed)} uploader-disjoint audio clips and manifest at {root}")


if __name__ == "__main__":
    main()
