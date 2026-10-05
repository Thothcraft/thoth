"""Local live-event fanout — feeds the node's SSE stream (contract §5).

One process-local hub; producers publish ``(kind, data)`` tuples and any
number of consumers hold bounded queues. Slow consumers shed the oldest
events rather than backpressuring producers — the hub carries UI/live
data, never the uplink path (observation durability lives in
``ObservationSpool``).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Tuple


class EventHub:
    def __init__(self, max_per_subscriber: int = 256):
        self._subs: List["queue.Queue"] = []
        self._lock = threading.Lock()
        self._max = max_per_subscriber
        self._seq = 0

    def publish(self, kind: str, data: Dict[str, Any]) -> int:
        """Broadcast one event; returns its stream sequence id."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            subs = list(self._subs)
        frame = {"id": seq, "ts": time.time(), "kind": kind, "data": data}
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Shed oldest, keep the newest — a wedged consumer must
                # never stall a producer.
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except Exception:
                    pass
        return seq

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=self._max)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def stream(self, q: "queue.Queue", keepalive_s: float = 15.0
               ) -> Iterator[Tuple[int, str]]:
        """Yield ``(seq, sse_line)`` until unsubscribed; emits a comment
        keepalive every ``keepalive_s`` so proxies hold the socket."""
        while True:
            try:
                frame = q.get(timeout=keepalive_s)
            except queue.Empty:
                yield 0, ": keepalive\n\n"
                continue
            yield int(frame["id"]), (
                f"id: {frame['id']}\n"
                f"event: {frame['kind']}\n"
                f"data: {json.dumps(frame['data'], separators=(',', ':'))}\n\n")


__all__ = ["EventHub"]
