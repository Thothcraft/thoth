import multiprocessing
import os
import sys
import time


def run_script1():
    """Replace this child with the original Example 2 GUI process."""
    os.execv(sys.executable, [sys.executable, "location_gui.py"])


def run_script2():
    """Replace this child with the original radar streamer process."""
    os.execv(sys.executable, [sys.executable, "run_udp_streaming_track.py"])


def stop_process(process):
    """Stop one Example 2 child without leaving the SPI radar open."""
    if process is None or not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def main():
    gui = multiprocessing.Process(target=run_script1, name="Example2GUI")
    streamer = None
    gui.start()

    try:
        # Preserve the original five-second GUI startup sequence, but do not
        # start the radar if the GUI import or display initialization fails.
        startup_deadline = time.monotonic() + 5
        while time.monotonic() < startup_deadline:
            if not gui.is_alive():
                gui.join()
                return gui.exitcode or 1
            time.sleep(0.1)

        streamer = multiprocessing.Process(target=run_script2, name="Example2Radar")
        streamer.start()

        # The old implementation joined both children forever. If the GUI was
        # closed or crashed, the streamer kept the radar open and the launcher
        # could not restore Thoth. End the session when either original module
        # exits and let finally close the other one.
        while gui.is_alive() and streamer.is_alive():
            time.sleep(0.2)

        gui.join(timeout=0.1)
        streamer.join(timeout=0.1)
        if gui.exitcode is not None:
            return gui.exitcode
        if streamer.exitcode is not None:
            return streamer.exitcode
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(streamer)
        stop_process(gui)


if __name__ == "__main__":
    raise SystemExit(main())
