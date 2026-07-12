import signal
import subprocess
import sys
import time


def terminate(process, timeout=3):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


if __name__ == "__main__":
    gui_process = None
    streamer_process = None

    def handle_signal(signum, frame):
        terminate(streamer_process)
        terminate(gui_process)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        gui_process = subprocess.Popen([sys.executable, "location_gui_advanced.py"])
        time.sleep(3)
        streamer_process = subprocess.Popen([sys.executable, "run_udp_streaming_track.py"])

        while True:
            gui_status = gui_process.poll()
            streamer_status = streamer_process.poll()

            if gui_status is not None:
                sys.exit(gui_status)
            if streamer_status is not None:
                terminate(gui_process)
                sys.exit(streamer_status)

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        terminate(streamer_process)
        terminate(gui_process)
