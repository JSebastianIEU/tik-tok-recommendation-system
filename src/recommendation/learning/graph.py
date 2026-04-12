from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .temporal import parse_dt


GRAPH_VERSION = "creator_video_dna_graph.v2"


def _write_table(frame: pd.DataFrame, output_stem: Path) -> Dict[str, str]:
    parquet_path = output_stem.with_suffix(".parquet")
    jsonl_path = output_stem.with_suffix(".jsonl")
    try:
        frame.to_parquet(parquet_path, index=False)
        return {"format": "parquet", "path": parquet_path.name}
    except Exception:
        records = frame.to_dict(orient="records")
        jsonl_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
            encoding="utf-8",
        )
        return {"format": "jsonl", "path": jsonl_path.name}


def _read_table(base_dir: Path, payload: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    table_meta = payload.get(stem)
    if isinstance(table_meta, dict):
        fmt = str(table_meta.get("format") or "").lower()
        path_name = str(table_meta.get("path") or "")
        path = (base_dir / path_name) if path_name else None
        if path is not None and path.exists():
            if fmt == "parquet":
                return pd.read_parquet(path).to_dict(orient="records")
            rows: List[Dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
            return rows
    # backward-compatible direct parquet lookup
    fallback_parquet = base_dir / f"{stem}.parquet"
    if fallback_parquet.exists():
        return pd.read_parquet(fallback_parquet).to_dict(orient="records")
    fallback_jsonl = base_dir / f"{stem}.jsonl"
    if fallback_jsonl.exists():
        rows: List[Dict[str, Any]] = []
        for line in fallback_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows
    return []


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_utc_iso(value: Any) -> Optional[str]:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(out):
        return out
    return default


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _extract_hashtags(row: Dict[str, Any]) -> List[str]:
    raw: List[str] = []
    hashtags = row.get("hashtags")
    if isinstance(hashtags, list):
        raw.extend(str(item) for item in hashtags if str(item).strip())
    for key in ("_runtime_text", "caption", "topic_key"):
        text = _coerce_text(row.get(key))
        if text:
            raw.extend(token for token in text.split() if token.strip().startswith("#"))
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("#"):
            cleaned = f"#{cleaned}"
        if len(cleaned) < 2:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out[:12]


def _extract_style_tags(row: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    fabric = row.get("_fabric_output")
    if isinstance(fabric, dict):
        visual = fabric.get("visual")
        if isinstance(visual, dict):
            raw = visual.get("style_tags")
            if isinstance(raw, list):
                tags.extend(str(item) for item in raw if str(item).strip())
    content_type = _coerce_text(row.get("content_type")).lower()
    if content_type:
        tags.append(content_type)
    out: List[str] = []
    seen: set[str] = set()
    for item in tags:
        cleaned = item.strip().lower().replace(" ", "_")
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out[:8]


def _audio_motif(row: Dict[str, Any]) -> Optional[str]:
    fabric = row.get("_fabric_output")
    audio = fabric.get("audio") if isinstance(fabric, dict) else None
    if not isinstance(audio, dict):
        return None
    speech = _coerce_float(audio.get("speech_ratio"), 0.0)
    tempo = _coerce_float(audio.get("tempo"), 0.0)
    energy = _coerce_float(audio.get("energy"), 0.0)
    music = bool(audio.get("music_presence", False))

    def _bucket(value: float, lo: float, hi: float) -> str:
        if value < lo:
            return "low"
        if value > hi:
            return "high"
        return "mid"

    speech_bucket = _bucket(speech, 0.35, 0.70)
    tempo_bucket = _bucket(tempo, 90.0, 140.0)
    energy_bucket = _bucket(energy, 0.35, 0.70)
    return (
        f"speech:{speech_bucket}|tempo:{tempo_bucket}|energy:{energy_bucket}|"
        f"music:{'yes' if music else 'no'}"
    )


def _creator_id(row: Dict[str, Any]) -> str:
    return _coerce_text(row.get("author_id")) or "unknown"


def _video_id(row: Dict[str, Any]) -> str:
    value = _coerce_text(row.get("video_id"))
    if value:
        return value
    row_id = _coerce_text(row.get("row_id"))
    if "::" in row_id:
        return row_id.split("::", 1)[0]
    return row_id or "unknown-video"


def _event_time(row: Dict[str, Any]) -> Optional[datetime]:
    return parse_dt(row.get("event_time")) or parse_dt(row.get("as_of_time"))


def _ingested_time(row: Dict[str, Any]) -> Optional[datetime]:
    return (
        parse_dt(row.get("ingested_at"))
        or parse_dt(row.get("scraped_at"))
        or _event_time(row)
    )


def _node_key(node_type: str, value: str) -> str:
    return f"{node_type}:{value}"


def _edge_key(source: str, target: str, edge_type: str) -> Tuple[str, str, str]:
    if source <= target:
        return source, target, edge_type
    return target, source, edge_type


@dataclass
class GraphBuildConfig:
    embedding_dim: int = 32
    walk_length: int = 12
    num_walks: int = 20
    context_size: int = 4
    seed: int = 13
    recency_half_life_days: float = 45.0
    include_creator_similarity: bool = True
    creator_similarity_top_k: int = 5
    creator_similarity_min_jaccard: float = 0.15

    def __post_init__(self) -> None:
        self.embedding_dim = max(4, int(self.embedding_dim))
        self.walk_length = max(4, int(self.walk_length))
        self.num_walks = max(2, int(self.num_walks))
        self.context_size = max(1, int(self.context_size))
        self.seed = int(self.seed)
        self.recency_half_life_days = max(1.0, float(self.recency_half_life_days))
        self.creator_similarity_top_k = max(1, int(self.creator_similarity_top_k))
        self.creator_similarity_min_jaccard = _clip(
            float(self.creator_similarity_min_jaccard), 0.0, 1.0
        )


@dataclass
class GraphBundle:
    version: str
    graph_bundle_id: str
    graph_schema_hash: str
    config: GraphBuildConfig
    created_at: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    embeddings_by_node: Dict[str, List[float]]
    video_embeddings: Dict[str, List[float]]
    creator_embeddings: Dict[str, List[float]]
    hashtag_embeddings: Dict[str, List[float]]
    audio_embeddings: Dict[str, List[float]]
    style_embeddings: Dict[str, List[float]]
    creator_neighbor_strength: Dict[str, float]
    video_dna: Dict[str, Dict[str, Any]]

    def query_embedding(self, query_row: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        video_id = _video_id(query_row)
        creator_id = _creator_id(query_row)
        hashtags = _extract_hashtags(query_row)
        style_tags = _extract_style_tags(query_row)
        motif = _audio_motif(query_row)

        vectors: List[np.ndarray] = []
        trace = {
            "video_id": video_id,
            "creator_id": creator_id,
            "hashtags": hashtags[:6],
            "style_tags": style_tags[:6],
            "audio_motif": motif,
            "sources": [],
        }
        raw_video = self.video_embeddings.get(video_id)
        if raw_video:
            vectors.append(np.asarray(raw_video, dtype=np.float32))
            trace["sources"].append("video")
        raw_creator = self.creator_embeddings.get(creator_id)
        if raw_creator:
            vectors.append(np.asarray(raw_creator, dtype=np.float32))
            trace["sources"].append("creator")
        for tag in hashtags:
            raw = self.hashtag_embeddings.get(tag)
            if raw:
                vectors.append(np.asarray(raw, dtype=np.float32))
                trace["sources"].append("hashtag")
        for tag in style_tags:
            raw = self.style_embeddings.get(tag)
            if raw:
                vectors.append(np.asarray(raw, dtype=np.float32))
                trace["sources"].append("style")
        if motif:
            raw = self.audio_embeddings.get(motif)
            if raw:
                vectors.append(np.asarray(raw, dtype=np.float32))
                trace["sources"].append("audio_motif")
        if not vectors:
            dim = max(1, int(self.config.embedding_dim))
            return np.zeros((dim,), dtype=np.float32), trace
        matrix = np.vstack(vectors).astype(np.float32)
        out = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.astype(np.float32), trace

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_frame = pd.DataFrame(self.nodes)
        edges_frame = pd.DataFrame(self.edges)
        embeddings_rows = [
            {
                "node_id": node_id,
                "node_type": node_id.split(":", 1)[0] if ":" in node_id else "unknown",
                "embedding": json.dumps(vector, ensure_ascii=False),
            }
            for node_id, vector in sorted(self.embeddings_by_node.items())
        ]
        embeddings_frame = pd.DataFrame(embeddings_rows)
        nodes_table = _write_table(nodes_frame, output_dir / "nodes")
        edges_table = _write_table(edges_frame, output_dir / "edges")
        embeddings_table = _write_table(embeddings_frame, output_dir / "embeddings")

        payload = {
            "version": self.version,
            "graph_bundle_id": self.graph_bundle_id,
            "graph_schema_hash": self.graph_schema_hash,
            "created_at": self.created_at,
            "config": asdict(self.config),
            "tables": {
                "nodes": nodes_table,
                "edges": edges_table,
                "embeddings": embeddings_table,
            },
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "video_embedding_count": len(self.video_embeddings),
            "creator_embedding_count": len(self.creator_embeddings),
            "hashtag_embedding_count": len(self.hashtag_embeddings),
            "audio_embedding_count": len(self.audio_embeddings),
            "style_embedding_count": len(self.style_embeddings),
        }
        (output_dir / "graph_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "bundle.pkl").open("wb") as fh:
            import pickle

            pickle.dump(self, fh)
        return output_dir / "graph_manifest.json"

    @classmethod
    def load(cls, output_dir: Path) -> "GraphBundle":
        bundle_pickle = output_dir / "bundle.pkl"
        if bundle_pickle.exists():
            with bundle_pickle.open("rb") as fh:
                import pickle

                loaded = pickle.load(fh)
            if isinstance(loaded, GraphBundle):
                return loaded

        manifest = json.loads((output_dir / "graph_manifest.json").read_text(encoding="utf-8"))
        tables = manifest.get("tables") if isinstance(manifest.get("tables"), dict) else {}
        nodes = _read_table(output_dir, dict(tables), "nodes")
        edges = _read_table(output_dir, dict(tables), "edges")
        embeddings_rows = _read_table(output_dir, dict(tables), "embeddings")
        embeddings_by_node: Dict[str, List[float]] = {}
        for row in embeddings_rows:
            node_id = str(row.get("node_id") or "")
            if not node_id:
                continue
            raw = row.get("embedding")
            if isinstance(raw, str) and raw.strip():
                try:
                    values = json.loads(raw)
                except json.JSONDecodeError:
                    values = []
            else:
                values = []
            embeddings_by_node[node_id] = [float(item) for item in list(values or [])]

        def _typed(prefix: str) -> Dict[str, List[float]]:
            out: Dict[str, List[float]] = {}
            for node_id, vector in embeddings_by_node.items():
                if not node_id.startswith(f"{prefix}:"):
                    continue
                out[node_id.split(":", 1)[1]] = vector
            return out

        return cls(
            version=str(manifest.get("version") or GRAPH_VERSION),
            graph_bundle_id=str(manifest.get("graph_bundle_id") or output_dir.name),
            graph_schema_hash=str(manifest.get("graph_schema_hash") or ""),
            config=GraphBuildConfig(**dict(manifest.get("config") or {})),
            created_at=str(manifest.get("created_at") or _to_utc_iso(datetime.now(timezone.utc))),
            nodes=nodes,
            edges=edges,
            embeddings_by_node=embeddings_by_node,
            video_embeddings=_typed("video"),
            creator_embeddings=_typed("creator"),
            hashtag_embeddings=_typed("hashtag"),
            audio_embeddings=_typed("audio_motif"),
            style_embeddings=_typed("style_signature"),
            creator_neighbor_strength={},
            video_dna={},
        )


def _node2vec_like_embeddings(
    adjacency: Dict[str, List[Tuple[str, float]]],
    *,
    config: GraphBuildConfig,
) -> Dict[str, List[float]]:
    node_ids = sorted(adjacency.keys())
    if not node_ids:
        return {}
    if len(node_ids) == 1:
        return {node_ids[0]: [1.0] + [0.0] * max(0, config.embedding_dim - 1)}

    rng = random.Random(config.seed)
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    co_counts: Dict[Tuple[int, int], float] = {}

    for start in node_ids:
        neighbors = adjacency.get(start, [])
        if not neighbors:
            continue
        for _ in range(config.num_walks):
            walk = [start]
            current = start
            for _step in range(max(1, config.walk_length - 1)):
                next_candidates = adjacency.get(current, [])
                if not next_candidates:
                    break
                choices = [item[0] for item in next_candidates]
                weights = [max(1e-6, float(item[1])) for item in next_candidates]
                current = rng.choices(choices, weights=weights, k=1)[0]
                walk.append(current)
            for center_idx, center in enumerate(walk):
                left = max(0, center_idx - config.context_size)
                right = min(len(walk), center_idx + config.context_size + 1)
                center_id = index[center]
                for ctx_idx in range(left, right):
                    if ctx_idx == center_idx:
                        continue
                    context_id = index[walk[ctx_idx]]
                    key = (center_id, context_id)
                    co_counts[key] = co_counts.get(key, 0.0) + 1.0

    if not co_counts:
        eye = np.eye(len(node_ids), dtype=np.float32)
        dim = min(config.embedding_dim, eye.shape[1])
        matrix = _normalize_rows(eye[:, :dim])
        return {node_ids[idx]: matrix[idx].tolist() for idx in range(len(node_ids))}

    from scipy import sparse as sp

    n = len(node_ids)
    rows_idx, cols_idx, vals = [], [], []
    for (center_id, context_id), value in co_counts.items():
        rows_idx.append(center_id)
        cols_idx.append(context_id)
        vals.append(float(value))
    matrix = sp.csr_matrix(
        (vals, (rows_idx, cols_idx)), shape=(n, n), dtype=np.float32
    )
    matrix = matrix + matrix.T
    row_sums = np.maximum(matrix.sum(axis=1).A1, 1.0)
    matrix = sp.diags(1.0 / row_sums) @ matrix

    dim = min(config.embedding_dim, max(2, n - 1))
    svd = TruncatedSVD(n_components=dim, random_state=config.seed)
    reduced = svd.fit_transform(matrix).astype(np.float32)
    reduced = _normalize_rows(reduced)
    return {node_ids[idx]: reduced[idx].tolist() for idx in range(len(node_ids))}


def _creator_similarity_edges(
    creator_signatures: Dict[str, set[str]],
    *,
    config: GraphBuildConfig,
) -> List[Tuple[str, str, float]]:
    if not config.include_creator_similarity:
        return []
    creators = sorted(creator_signatures.keys())
    out: List[Tuple[str, str, float]] = []
    for source in creators:
        scores: List[Tuple[str, float]] = []
        source_set = creator_signatures.get(source, set())
        if not source_set:
            continue
        for target in creators:
            if source == target:
                continue
            target_set = creator_signatures.get(target, set())
            if not target_set:
                continue
            inter = len(source_set & target_set)
            union = len(source_set | target_set)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard < config.creator_similarity_min_jaccard:
                continue
            scores.append((target, float(jaccard)))
        scores.sort(key=lambda item: item[1], reverse=True)
        for target, score in scores[: config.creator_similarity_top_k]:
            out.append((source, target, score))
    return out


def build_creator_video_dna_graph(
    rows: Sequence[Dict[str, Any]],
    *,
    as_of_time: Optional[Any] = None,
    run_cutoff_time: Optional[Any] = None,
    config: Optional[GraphBuildConfig] = None,
) -> GraphBundle:
    cfg = config or GraphBuildConfig()
    as_of = parse_dt(as_of_time)
    run_cutoff = parse_dt(run_cutoff_time) or as_of

    node_meta: Dict[str, Dict[str, Any]] = {}
    edge_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    creator_signatures: Dict[str, set[str]] = {}
    creator_weight_degree: Dict[str, float] = {}
    video_dna: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        event_time = _event_time(row)
        ingested_at = _ingested_time(row)
        if as_of is not None and event_time is not None and event_time > as_of:
            continue
        if run_cutoff is not None and ingested_at is not None and ingested_at > run_cutoff:
            continue
        video_id = _video_id(row)
        creator_id = _creator_id(row)
        hashtags = _extract_hashtags(row)
        style_tags = _extract_style_tags(row)
        motif = _audio_motif(row)

        recency_weight = 1.0
        if run_cutoff is not None and event_time is not None:
            age_days = max(0.0, (run_cutoff - event_time).total_seconds() / 86400.0)
            recency_weight = 2 ** (-(age_days / cfg.recency_half_life_days))
        features = row.get("features", {})
        if not isinstance(features, dict):
            features = {}
        missingness = features.get("missingness_flags")
        missing_count = len(missingness) if isinstance(missingness, list) else 0
        quality_weight = _clip(1.0 - (0.05 * missing_count), 0.5, 1.0)
        interaction_weight = 1.0 + (
            0.15
            * math.log1p(
                float(len(hashtags))
                + float(len(style_tags))
                + float(_coerce_float(features.get("keyword_count"), 0.0))
            )
        )
        edge_weight = float(recency_weight * quality_weight * interaction_weight)

        creator_node = _node_key("creator", creator_id)
        video_node = _node_key("video", video_id)
        node_meta.setdefault(creator_node, {"node_id": creator_node, "node_type": "creator", "value": creator_id})
        node_meta.setdefault(video_node, {"node_id": video_node, "node_type": "video", "value": video_id})
        key = _edge_key(creator_node, video_node, "creator_video")
        edge_meta[key] = {
            "source": key[0],
            "target": key[1],
            "edge_type": key[2],
            "weight": edge_meta.get(key, {}).get("weight", 0.0) + edge_weight,
        }
        creator_weight_degree[creator_id] = creator_weight_degree.get(creator_id, 0.0) + edge_weight
        signature = creator_signatures.setdefault(creator_id, set())

        for tag in hashtags:
            tag_node = _node_key("hashtag", tag)
            node_meta.setdefault(tag_node, {"node_id": tag_node, "node_type": "hashtag", "value": tag})
            tag_key = _edge_key(video_node, tag_node, "video_hashtag")
            edge_meta[tag_key] = {
                "source": tag_key[0],
                "target": tag_key[1],
                "edge_type": tag_key[2],
                "weight": edge_meta.get(tag_key, {}).get("weight", 0.0) + edge_weight,
            }
            signature.add(f"tag::{tag}")

        if motif:
            motif_node = _node_key("audio_motif", motif)
            node_meta.setdefault(
                motif_node,
                {"node_id": motif_node, "node_type": "audio_motif", "value": motif},
            )
            motif_key = _edge_key(video_node, motif_node, "video_audio_motif")
            edge_meta[motif_key] = {
                "source": motif_key[0],
                "target": motif_key[1],
                "edge_type": motif_key[2],
                "weight": edge_meta.get(motif_key, {}).get("weight", 0.0) + edge_weight,
            }
            signature.add(f"audio::{motif}")

        for tag in style_tags:
            style_node = _node_key("style_signature", tag)
            node_meta.setdefault(
                style_node,
                {"node_id": style_node, "node_type": "style_signature", "value": tag},
            )
            style_key = _edge_key(video_node, style_node, "video_style_signature")
            edge_meta[style_key] = {
                "source": style_key[0],
                "target": style_key[1],
                "edge_type": style_key[2],
                "weight": edge_meta.get(style_key, {}).get("weight", 0.0) + edge_weight,
            }
            signature.add(f"style::{tag}")

        video_dna[video_id] = {
            "creator_id": creator_id,
            "hashtags": hashtags,
            "style_tags": style_tags,
            "audio_motif": motif,
        }

    for source, target, score in _creator_similarity_edges(creator_signatures, config=cfg):
        source_node = _node_key("creator", source)
        target_node = _node_key("creator", target)
        node_meta.setdefault(source_node, {"node_id": source_node, "node_type": "creator", "value": source})
        node_meta.setdefault(target_node, {"node_id": target_node, "node_type": "creator", "value": target})
        key = _edge_key(source_node, target_node, "creator_similarity")
        edge_meta[key] = {
            "source": key[0],
            "target": key[1],
            "edge_type": key[2],
            "weight": max(edge_meta.get(key, {}).get("weight", 0.0), float(score)),
        }

    nodes = sorted(node_meta.values(), key=lambda item: str(item.get("node_id")))
    edges = sorted(edge_meta.values(), key=lambda item: (str(item.get("source")), str(item.get("target")), str(item.get("edge_type"))))
    adjacency: Dict[str, List[Tuple[str, float]]] = {str(node["node_id"]): [] for node in nodes}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        weight = max(1e-6, float(edge.get("weight") or 0.0))
        adjacency.setdefault(source, []).append((target, weight))
        adjacency.setdefault(target, []).append((source, weight))

    embeddings_by_node = _node2vec_like_embeddings(adjacency, config=cfg)
    video_embeddings: Dict[str, List[float]] = {}
    creator_embeddings: Dict[str, List[float]] = {}
    hashtag_embeddings: Dict[str, List[float]] = {}
    audio_embeddings: Dict[str, List[float]] = {}
    style_embeddings: Dict[str, List[float]] = {}
    for node_id, vector in embeddings_by_node.items():
        if ":" not in node_id:
            continue
        node_type, value = node_id.split(":", 1)
        if node_type == "video":
            video_embeddings[value] = vector
        elif node_type == "creator":
            creator_embeddings[value] = vector
        elif node_type == "hashtag":
            hashtag_embeddings[value] = vector
        elif node_type == "audio_motif":
            audio_embeddings[value] = vector
        elif node_type == "style_signature":
            style_embeddings[value] = vector

    max_creator_degree = max(creator_weight_degree.values()) if creator_weight_degree else 1.0
    creator_neighbor_strength = {
        creator_id: float(round(weight / max(1e-6, max_creator_degree), 6))
        for creator_id, weight in creator_weight_degree.items()
    }

    schema_hash = _stable_hash(
        {
            "version": GRAPH_VERSION,
            "node_types": sorted({node["node_type"] for node in nodes}),
            "edge_types": sorted({edge["edge_type"] for edge in edges}),
            "config": asdict(cfg),
        }
    )
    bundle_id = _stable_hash(
        {
            "schema_hash": schema_hash,
            "node_hash": _stable_hash(nodes),
            "edge_hash": _stable_hash(edges),
            "embedding_hash": _stable_hash(
                {key: value[:8] for key, value in sorted(embeddings_by_node.items())}
            ),
        }
    )[:16]
    created_at = _to_utc_iso(datetime.now(timezone.utc)) or ""
    return GraphBundle(
        version=GRAPH_VERSION,
        graph_bundle_id=bundle_id,
        graph_schema_hash=schema_hash,
        config=cfg,
        created_at=created_at,
        nodes=nodes,
        edges=edges,
        embeddings_by_node=embeddings_by_node,
        video_embeddings=video_embeddings,
        creator_embeddings=creator_embeddings,
        hashtag_embeddings=hashtag_embeddings,
        audio_embeddings=audio_embeddings,
        style_embeddings=style_embeddings,
        creator_neighbor_strength=creator_neighbor_strength,
        video_dna=video_dna,
    )


def annotate_rows_with_graph_features(
    rows: Iterable[Dict[str, Any]],
    graph_bundle: Optional[GraphBundle],
) -> None:
    if graph_bundle is None:
        return
    for row in rows:
        features = row.get("features")
        if not isinstance(features, dict):
            continue
        author_id = _creator_id(row)
        video_id = _video_id(row)
        dna = graph_bundle.video_dna.get(video_id, {})
        features["graph_creator_neighbor_strength"] = float(
            graph_bundle.creator_neighbor_strength.get(author_id, 0.0)
        )
        features["graph_hashtag_count"] = float(
            len(dna.get("hashtags", [])) if isinstance(dna, dict) else 0
        )
        features["graph_style_count"] = float(
            len(dna.get("style_tags", [])) if isinstance(dna, dict) else 0
        )
        features["graph_has_audio_motif"] = float(
            1.0 if isinstance(dna, dict) and dna.get("audio_motif") else 0.0
        )
