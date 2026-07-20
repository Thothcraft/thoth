import importlib.util
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "WS/MMW-HAT/MMW-HAT-Release/example_2_track/run_example_track.py"
)


class FakeProcess:
    def __init__(self, alive, exitcode):
        self._alive = alive
        self.exitcode = exitcode
        self.started = False
        self.terminated = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self._alive = False
        self.exitcode = -9


def load_runner():
    spec = importlib.util.spec_from_file_location("example2_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Example2RunnerTests(unittest.TestCase):
    def test_gui_startup_failure_does_not_start_radar(self):
        runner = load_runner()
        gui = FakeProcess(alive=False, exitcode=7)
        created = []

        def process_factory(*args, **kwargs):
            created.append(kwargs.get("name"))
            return gui

        with mock.patch.object(runner.multiprocessing, "Process", process_factory), \
             mock.patch.object(runner.time, "monotonic", return_value=0):
            self.assertEqual(runner.main(), 7)

        self.assertEqual(created, ["Example2GUI"])

    def test_streamer_exit_stops_the_gui(self):
        runner = load_runner()
        gui = FakeProcess(alive=True, exitcode=None)
        streamer = FakeProcess(alive=False, exitcode=9)
        processes = iter([gui, streamer])

        with mock.patch.object(
            runner.multiprocessing, "Process", side_effect=lambda *args, **kwargs: next(processes)
        ), mock.patch.object(runner.time, "monotonic", side_effect=[0, 6]):
            self.assertEqual(runner.main(), 9)

        self.assertTrue(gui.terminated)


if __name__ == "__main__":
    unittest.main()
