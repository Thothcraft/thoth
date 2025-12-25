import multiprocessing
import time
import subprocess


def run_script1():
    # Start the first Python script
    subprocess.run(["python", "run_vis.py"])


def run_script2():
    # Start the second Python script
    subprocess.run(["python", "run_udp_streaming_vis.py"])


if __name__ == "__main__":
    # Create a process for the first script
    process1 = multiprocessing.Process(target=run_script1)
    # Start the first process
    process1.start()

    # Wait for 20 seconds before starting the second script
    time.sleep(20)

    # Create a process for the second script
    process2 = multiprocessing.Process(target=run_script2)
    # Start the second process
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()
