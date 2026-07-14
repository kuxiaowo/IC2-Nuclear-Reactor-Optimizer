from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .engine import SimulationRun


class TraceStore:
    def __init__(self, root: Path | str = ".data/traces") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, simulation_id: str) -> Path:
        return self.root / f"{simulation_id}.h5"

    def write(self, simulation_id: str, run: SimulationRun) -> Path:
        path = self.path_for(simulation_id)
        records = run.records
        count = len(records)
        slot_count = len(records[0].component_ids) if records and records[0].component_ids else 0
        chunk = min(max(1, count), 1024)
        with h5py.File(path, "w") as handle:
            handle.attrs["summary_json"] = run.summary.model_dump_json()
            handle.attrs["reactor_ticks"] = count
            scalar_fields = {
                "hull_heat": np.asarray([r.hull_heat for r in records], dtype=np.int64),
                "max_hull_heat": np.asarray([r.max_hull_heat for r in records], dtype=np.int64),
                "eu_per_tick": np.asarray([r.eu_per_tick for r in records], dtype=np.float64),
                "total_eu": np.asarray([r.total_eu for r in records], dtype=np.float64),
                "generated_heat": np.asarray([r.generated_heat for r in records], dtype=np.int64),
                "vented_heat": np.asarray([r.vented_heat for r in records], dtype=np.int64),
            }
            for name, values in scalar_fields.items():
                handle.create_dataset(name, data=values, chunks=(chunk,), compression="gzip", shuffle=True)
            if slot_count:
                string_type = h5py.string_dtype("utf-8")
                ids = np.asarray([r.component_ids for r in records], dtype=object)
                handle.create_dataset("component_ids", data=ids, dtype=string_type, chunks=(chunk, slot_count), compression="gzip")
                handle.create_dataset(
                    "component_heat",
                    data=np.asarray([r.component_heat for r in records], dtype=np.int32),
                    chunks=(chunk, slot_count), compression="gzip", shuffle=True,
                )
                handle.create_dataset(
                    "component_damage",
                    data=np.asarray([r.component_damage for r in records], dtype=np.int32),
                    chunks=(chunk, slot_count), compression="gzip", shuffle=True,
                )
        return path

    def _record_index(self, game_tick: int, count: int) -> int:
        if count <= 0:
            raise IndexError("轨迹为空")
        return min(count - 1, max(0, game_tick // 20))

    def page(self, simulation_id: str, offset: int, limit: int) -> dict:
        path = self.path_for(simulation_id)
        with h5py.File(path, "r") as handle:
            count = int(handle.attrs["reactor_ticks"])
            total_game_ticks = count * 20
            start = min(max(0, offset), total_game_ticks)
            end = min(total_game_ticks, start + min(max(1, limit), 10_000))
            rows = []
            for game_tick in range(start, end):
                idx = self._record_index(game_tick, count)
                rows.append({
                    "game_tick": game_tick + 1,
                    "seconds": (game_tick + 1) / 20.0,
                    "reactor_tick": idx + 1,
                    "hull_heat": int(handle["hull_heat"][idx]),
                    "max_hull_heat": int(handle["max_hull_heat"][idx]),
                    "eu_per_tick": float(handle["eu_per_tick"][idx]),
                    "total_eu": float(handle["total_eu"][idx]) - float(handle["eu_per_tick"][idx]) * (19 - game_tick % 20),
                    "generated_heat": int(handle["generated_heat"][idx]) if game_tick % 20 == 19 else 0,
                    "vented_heat": int(handle["vented_heat"][idx]) if game_tick % 20 == 19 else 0,
                })
            return {"offset": start, "limit": limit, "total": total_game_ticks, "rows": rows}

    def components_at(self, simulation_id: str, game_tick: int) -> dict:
        path = self.path_for(simulation_id)
        with h5py.File(path, "r") as handle:
            count = int(handle.attrs["reactor_ticks"])
            idx = self._record_index(game_tick, count)
            if "component_ids" not in handle:
                return {"game_tick": game_tick, "reactor_tick": idx + 1, "components": []}
            ids = [item.decode() if isinstance(item, bytes) else str(item) for item in handle["component_ids"][idx]]
            heat = handle["component_heat"][idx]
            damage = handle["component_damage"][idx]
            return {
                "game_tick": game_tick,
                "reactor_tick": idx + 1,
                "components": [
                    {"slot": i, "component_id": component_id, "heat": int(heat[i]), "damage": int(damage[i])}
                    for i, component_id in enumerate(ids)
                ],
            }

    def chart(self, simulation_id: str, points: int = 1000) -> dict:
        path = self.path_for(simulation_id)
        with h5py.File(path, "r") as handle:
            count = int(handle.attrs["reactor_ticks"])
            if count == 0:
                return {"points": []}
            target = min(max(10, points), 5000)
            indices = np.unique(np.linspace(0, count - 1, target, dtype=np.int64))
            return {
                "points": [
                    {
                        "game_tick": int(index + 1) * 20,
                        "hull_heat": int(handle["hull_heat"][index]),
                        "max_hull_heat": int(handle["max_hull_heat"][index]),
                        "eu_per_tick": float(handle["eu_per_tick"][index]),
                        "total_eu": float(handle["total_eu"][index]),
                    }
                    for index in indices
                ]
            }

    def csv_rows(self, simulation_id: str, include_components: bool = False) -> Iterator[str]:
        output = io.StringIO()
        writer = csv.writer(output)
        header = ["game_tick", "seconds", "reactor_tick", "hull_heat", "max_hull_heat", "eu_per_tick", "total_eu", "generated_heat", "vented_heat"]
        with h5py.File(self.path_for(simulation_id), "r") as handle:
            count = int(handle.attrs["reactor_ticks"])
            slots = handle["component_heat"].shape[1] if include_components and "component_heat" in handle else 0
            if slots:
                for slot in range(slots):
                    header.extend((f"slot_{slot + 1}_id", f"slot_{slot + 1}_heat", f"slot_{slot + 1}_damage"))
            writer.writerow(header)
            yield output.getvalue()
            output.seek(0); output.truncate(0)
            for game_tick in range(count * 20):
                idx = game_tick // 20
                row = [
                    game_tick + 1, (game_tick + 1) / 20.0, idx + 1,
                    int(handle["hull_heat"][idx]), int(handle["max_hull_heat"][idx]),
                    float(handle["eu_per_tick"][idx]),
                    float(handle["total_eu"][idx]) - float(handle["eu_per_tick"][idx]) * (19 - game_tick % 20),
                    int(handle["generated_heat"][idx]) if game_tick % 20 == 19 else 0,
                    int(handle["vented_heat"][idx]) if game_tick % 20 == 19 else 0,
                ]
                if slots:
                    ids = handle["component_ids"][idx]
                    heats = handle["component_heat"][idx]
                    damages = handle["component_damage"][idx]
                    for slot in range(slots):
                        item = ids[slot].decode() if isinstance(ids[slot], bytes) else str(ids[slot])
                        row.extend((item, int(heats[slot]), int(damages[slot])))
                writer.writerow(row)
                if game_tick % 1000 == 999:
                    yield output.getvalue()
                    output.seek(0); output.truncate(0)
            if output.tell():
                yield output.getvalue()

