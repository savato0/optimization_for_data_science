from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively plot a projected FW trace.")
    parser.add_argument("trace_json", help="Projected trace JSON produced by run_fw_experiments.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    points = payload.get("points", [])
    if not isinstance(points, list) or not points:
        raise ValueError(f"No projected trace points found in {args.trace_json}.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("matplotlib is required to plot projected FW traces.") from exc

    plotter = TracePlotter(payload, points, plt)
    plotter.show()


class TracePlotter:
    def __init__(self, payload: dict[str, Any], points: list[dict[str, Any]], plt: Any) -> None:
        self.payload = payload
        self.points = points
        self.plt = plt
        self.index = 0
        self.xs = [float(point["coord_1"]) for point in points]
        self.ys = [float(point["coord_2"]) for point in points]

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.path_line = None
        self.current_point = None
        self.info_text = None

    def show(self) -> None:
        self._draw_static()
        self._update()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.plt.show()

    def _draw_static(self) -> None:
        self.ax.plot(self.xs, self.ys, color="0.8", linewidth=1.0, label="FW trace")
        self.path_line, = self.ax.plot([], [], color="#1f77b4", linewidth=1.8)
        self.current_point = self.ax.scatter([], [], s=80, color="#d62728", zorder=5)

        projection = self.payload.get("projection", {})
        if isinstance(projection, dict):
            self._draw_marker(projection.get("origin"), marker="o", color="#2ca02c")
            self._draw_marker(projection.get("reference"), marker="*", color="#9467bd")
            self._draw_marker(projection.get("xu"), marker="X", color="#ff7f0e")

        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        self.ax.set_xlabel(str(projection.get("axis_1_label", "coord_1")))
        self.ax.set_ylabel(str(projection.get("axis_2_label", "coord_2")))
        self.ax.set_title(
            f"{self.payload.get('matrix')} / {self.payload.get('vector')} / "
            f"{self.payload.get('x0_key')}"
        )
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best")

    def _draw_marker(
        self,
        marker_payload: Any,
        *,
        marker: str,
        color: str,
    ) -> None:
        if not isinstance(marker_payload, dict):
            return
        self.ax.scatter(
            [float(marker_payload["coord_1"])],
            [float(marker_payload["coord_2"])],
            marker=marker,
            color=color,
            s=110,
            label=str(marker_payload.get("label", "")),
            zorder=4,
        )

    def _on_key(self, event: Any) -> None:
        if event.key in {"right", " ", "space"}:
            self.index = min(self.index + 1, len(self.points) - 1)
        elif event.key == "left":
            self.index = max(self.index - 1, 0)
        elif event.key == "home":
            self.index = 0
        elif event.key == "end":
            self.index = len(self.points) - 1
        else:
            return
        self._update()

    def _update(self) -> None:
        point = self.points[self.index]
        self.path_line.set_data(self.xs[: self.index + 1], self.ys[: self.index + 1])
        self.current_point.set_offsets([[self.xs[self.index], self.ys[self.index]]])
        self.info_text.set_text(
            "\n".join(
                [
                    f"trace point {self.index + 1}/{len(self.points)}",
                    f"iteration: {_format_value(point.get('iteration'))}",
                    f"objective: {_format_value(point.get('objective'))}",
                    f"fw_gap: {_format_value(point.get('fw_gap'))}",
                    f"rel_gap: {_format_value(point.get('relative_fw_gap'))}",
                    f"alpha: {_format_value(point.get('alpha'))}",
                    f"dist ref: {_format_value(point.get('distance_to_reference'))}",
                    f"dist x_u: {_format_value(point.get('distance_to_xu'))}",
                ]
            )
        )
        self.fig.canvas.draw_idle()


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


if __name__ == "__main__":
    main()
