import json
import math
import os
import random
import time

import pyarrow as pa
import sedona.db
import sedonadb

SIMRA_DIR = "simra"
RESULTS_DIR = "results"
BIN_NAME = "sedona"
SEED = 0x5EED_5EED_5EED_5EED
RUNS = 30
KNN_N = 5
PIP_HALF = 0.01
RANGE_HALF = 0.05
CAP = 100
DEPTH = 100


def iter_files(root):
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def parse_simra(root):
    points = []
    for path in iter_files(root):
        with open(path) as file:
            lines = file.read().splitlines()

        marker = next((i for i, line in enumerate(lines) if "=" * 10 in line), None)
        if marker is None:
            continue

        for line in lines[marker + 3 :]:
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                x = float(fields[0].strip())
                y = float(fields[1].strip())
            except ValueError:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            points.append((len(points), x, y))
    return points


def square_wkt(cx, cy, half):
    return (
        f"POLYGON(("
        f"{cx - half} {cy - half}, "
        f"{cx + half} {cy - half}, "
        f"{cx + half} {cy + half}, "
        f"{cx - half} {cy + half}, "
        f"{cx - half} {cy - half}))"
    )


def duration(seconds):
    secs, nanos = divmod(round(seconds * 1_000_000_000), 1_000_000_000)
    return {"secs": secs, "nanos": nanos}


def data_point(point_id, x, y):
    return {"id": point_id, "point": {"x": x, "y": y}}


KNN_SQL = (
    "SELECT p.id AS id, p.x AS x, p.y AS y FROM points p "
    "JOIN (SELECT ST_Point($1, $2) AS geom) q "
    "ON ST_KNN(q.geom, p.geom, $3, false) "
    "ORDER BY ST_Distance(p.geom, ST_Point($1, $2))"
)
KNN_BATCH_SQL = (
    "SELECT q.qid AS qid, p.id AS id, p.x AS x, p.y AS y FROM knn_queries q "
    "JOIN points p ON ST_KNN(q.geom, p.geom, $1, false) "
    "ORDER BY q.qid, ST_Distance(p.geom, q.geom)"
)
PIP_SQL = "SELECT ST_Contains(ST_GeomFromText($1), ST_Point($2, $3)) AS contained"
RANGE_SQL = "SELECT p.id AS id FROM points p WHERE ST_Intersects(p.geom, ST_GeomFromText($1))"


def save_report(report):
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    os.makedirs(result_dir, exist_ok=True)

    i = 0
    while os.path.exists(os.path.join(result_dir, f"benchmark_{BIN_NAME}_{i:02}.json")):
        i += 1
    path = os.path.join(result_dir, f"benchmark_{BIN_NAME}_{i:02}.json")

    with open(path, "w") as file:
        json.dump(report, file, indent=2)
    return path


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), SIMRA_DIR)
    sd = sedona.db.connect()

    full_start = time.perf_counter()

    parse_start = time.perf_counter()
    points = parse_simra(root)
    parse_time = time.perf_counter() - parse_start
    print(f"parsed {len(points)} simra points in {parse_time:.3f}s")

    if len(points) < RUNS:
        raise RuntimeError(f"expected at least {RUNS} points, found {len(points)}")

    table = pa.table(
        {
            "id": pa.array([p[0] for p in points], pa.int64()),
            "x": pa.array([p[1] for p in points], pa.float64()),
            "y": pa.array([p[2] for p in points], pa.float64()),
        }
    )

    build_start = time.perf_counter()
    sd.create_data_frame(table).to_view("points_raw", overwrite=True)
    sd.sql("SELECT id, x, y, ST_Point(x, y) AS geom FROM points_raw").to_memtable().to_view(
        "points", overwrite=True
    )
    build_time = time.perf_counter() - build_start
    print(f"built sedona table in {build_time:.3f}s")

    coords = {p[0]: (p[1], p[2]) for p in points}

    candidates = sorted(coords)
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:RUNS]

    source_points = []
    knn_results = []
    pip_results = []
    range_results = []

    for point_id in selected:
        x, y = coords[point_id]
        source_points.append(data_point(point_id, x, y))

        start = time.perf_counter()
        knn = sd.sql(KNN_SQL, params=(x, y, KNN_N)).to_arrow_table()
        knn_time = time.perf_counter() - start
        knn_points = [data_point(row["id"], row["x"], row["y"]) for row in knn.to_pylist()]
        assert len(knn_points) == KNN_N, f"expected {KNN_N} points, got {len(knn_points)}"
        knn_results.append(
            {"pointId": point_id, "dur": duration(knn_time), "points": knn_points}
        )

        start = time.perf_counter()
        contained = sd.sql(
            PIP_SQL, params=(square_wkt(x, y, PIP_HALF), x, y)
        ).to_arrow_table()
        contained_time = time.perf_counter() - start
        assert contained.column("contained")[0].as_py(), "pip not working"

        start = time.perf_counter()
        not_contained = sd.sql(
            PIP_SQL, params=(square_wkt(x + 4.0 * PIP_HALF, y, PIP_HALF), x, y)
        ).to_arrow_table()
        not_contained_time = time.perf_counter() - start
        assert not not_contained.column("contained")[0].as_py(), "pip not working"

        pip_results.append(
            {
                "pointId": point_id,
                "containedDur": duration(contained_time),
                "notContainedDur": duration(not_contained_time),
            }
        )

        start = time.perf_counter()
        sd.sql(RANGE_SQL, params=(square_wkt(x, y, RANGE_HALF),)).to_arrow_table()
        range_time = time.perf_counter() - start
        range_results.append({"pointId": point_id, "dur": duration(range_time)})

    query_table = pa.table(
        {
            "qid": pa.array(selected, pa.int64()),
            "x": pa.array([coords[i][0] for i in selected], pa.float64()),
            "y": pa.array([coords[i][1] for i in selected], pa.float64()),
        }
    )
    sd.create_data_frame(query_table).to_view("knn_queries_raw", overwrite=True)
    sd.sql("SELECT qid, ST_Point(x, y) AS geom FROM knn_queries_raw").to_memtable().to_view(
        "knn_queries", overwrite=True
    )

    batch_start = time.perf_counter()
    batch = sd.sql(KNN_BATCH_SQL, params=(KNN_N,)).to_arrow_table()
    batch_time = time.perf_counter() - batch_start

    batch_points = {}
    for row in batch.to_pylist():
        batch_points.setdefault(row["qid"], []).append(
            data_point(row["id"], row["x"], row["y"])
        )
    knn_batch_results = [
        {"pointId": point_id, "points": batch_points.get(point_id, [])}
        for point_id in selected
    ]

    full_time = time.perf_counter() - full_start

    report = {
        "type": "BuildQuad",
        "engine": f"sedonadb {sedonadb.__version__}",
        "data": {
            "full_runtime": duration(full_time),
            "cap": CAP,
            "depth": DEPTH,
            "data parse time": duration(parse_time),
            "sedona build time": duration(build_time),
            "num points": len(points),
            "table bytes": table.nbytes,
            "source points": source_points,
            "knn results": knn_results,
            "knn n": KNN_N,
            "knn batch runtime": duration(batch_time),
            "knn batch results": knn_batch_results,
            "pip results": pip_results,
            "range results": range_results,
        },
    }

    path = save_report(report)
    print(f"saved report to {path}")

    knn_total = sum(r["dur"]["secs"] * 1e9 + r["dur"]["nanos"] for r in knn_results)
    pip_total = sum(
        r["containedDur"]["secs"] * 1e9 + r["containedDur"]["nanos"]
        for r in pip_results
    )
    range_total = sum(r["dur"]["secs"] * 1e9 + r["dur"]["nanos"] for r in range_results)
    print(f"knn total:   {knn_total / 1e9:.3f}s")
    print(f"knn batch:   {batch_time:.3f}s")
    print(f"pip total:   {pip_total / 1e9:.3f}s")
    print(f"range total: {range_total / 1e9:.3f}s")
    print(f"full runtime: {full_time:.3f}s")


if __name__ == "__main__":
    main()
