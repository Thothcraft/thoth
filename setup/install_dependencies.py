#!/usr/bin/env python3
"""
Dependency Installation Script for Thoth Device

This script handles installation of all required dependencies for the Thoth device,
including platform-specific handling for PyAudio on macOS.

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
import platform
import os


def run_command(cmd, check=True, shell=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=check,
            shell=shell
        )
        if result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False, e.stderr


def check_brew_installed():
    """Check if Homebrew is installed on macOS."""
    try:
        subprocess.run(['brew', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_brew():
    """Install Homebrew on macOS."""
    print("Installing Homebrew...")
    install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    return run_command(install_cmd, shell=True)


def install_portaudio_macos():
    """Install PortAudio on macOS using Homebrew."""
    if not check_brew_installed():
        print("Homebrew not found. Installing...")
        success, _ = install_brew()
        if not success:
            print("Failed to install Homebrew. Please install it manually:")
            print('  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            return False
    
    print("Installing PortAudio via Homebrew...")
    success, _ = run_command(['brew', 'install', 'portaudio'], check=False)
    return success


def install_pyaudio_macos():
    """Install PyAudio on macOS with proper flags."""
    # First ensure PortAudio is installed
    if not install_portaudio_macos():
        print("Warning: Could not install PortAudio. PyAudio may fail to install.")
    
    # Determine Homebrew prefix (different for Intel vs Apple Silicon)
    try:
        result = subprocess.run(['brew', '--prefix'], capture_output=True, text=True, check=True)
        brew_prefix = result.stdout.strip()
    except:
        # Default paths
        if platform.machine() == 'arm64':
            brew_prefix = '/opt/homebrew'
        else:
            brew_prefix = '/usr/local'
    
    include_path = f"{brew_prefix}/include"
    lib_path = f"{brew_prefix}/lib"
    
    # Set environment variables and install PyAudio
    env = os.environ.copy()
    env['CFLAGS'] = f"-I{include_path} -L{lib_path}"
    env['LDFLAGS'] = f"-L{lib_path}"
    
    print(f"Installing PyAudio with CFLAGS=-I{include_path} -L{lib_path}")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'pyaudio'],
            env=env,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        # Try alternative method
        print("Trying alternative installation method...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 
                 '--global-option=build_ext',
                 f'--global-option=-I{include_path}',
                 f'--global-option=-L{lib_path}',
                 'pyaudio'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            print("Failed to install PyAudio. Please try manually:")
            print(f'  CFLAGS="-I{include_path} -L{lib_path}" pip install pyaudio')
            return False


def install_pyaudio_linux():
    """Install PyAudio on Linux."""
    # Try to install system dependencies first
    distro_cmds = [
        # Debian/Ubuntu
        ['sudo', 'apt-get', 'install', '-y', 'python3-pyaudio', 'portaudio19-dev'],
        # Fedora/RHEL
        ['sudo', 'dnf', 'install', '-y', 'python3-pyaudio', 'portaudio-devel'],
        # Arch
        ['sudo', 'pacman', '-S', '--noconfirm', 'python-pyaudio', 'portaudio'],
    ]
    
    for cmd in distro_cmds:
        success, _ = run_command(cmd, check=False)
        if success:
            break
    
    # Install via pip as fallback
    run_command([sys.executable, '-m', 'pip', 'install', 'pyaudio'], check=False)
    return True


def install_pyaudio_windows():
    """Install PyAudio on Windows."""
    # On Windows, pip install usually works directly
    success, _ = run_command([sys.executable, '-m', 'pip', 'install', 'pyaudio'], check=False)
    
    if not success:
        print("If PyAudio fails to install, download the wheel from:")
        print("  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        print("Then install with: pip install <downloaded_file>.whl")
    
    return success


def install_opencv():
    """Install OpenCV."""
    print("Installing OpenCV...")
    return run_command([sys.executable, '-m', 'pip', 'install', 'opencv-python'], check=False)


def install_base_dependencies():
    """Install base Python dependencies."""
    print("Installing base dependencies...")
    deps = [
        'flask',
        'flask-socketio',
        'flask-cors',
        'requests',
        'psutil',
        'netifaces',
        'apscheduler',
        'pyjwt',
        'python-dotenv',
    ]
    
    for dep in deps:
        run_command([sys.executable, '-m', 'pip', 'install', dep], check=False)
    
    return True


def main():
    """Main installation function."""
    print("=" * 60)
    print("Thoth Device Dependency Installer")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    
    # Install base dependencies
    install_base_dependencies()
    
    # Install OpenCV for camera support
    install_opencv()
    
    # Install PyAudio based on platform
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        print("\n--- Installing PyAudio for macOS ---")
        install_pyaudio_macos()
    elif system == 'Linux':
        print("\n--- Installing PyAudio for Linux ---")
        install_pyaudio_linux()
    elif system == 'Windows':
        print("\n--- Installing PyAudio for Windows ---")
        install_pyaudio_windows()
    else:
        print(f"Unknown platform: {system}")
        print("Please install PyAudio manually.")
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    
    # Verify installations
    print("\nVerifying installations...")
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("  ✗ OpenCV not installed")
    
    try:
        import pyaudio
        print(f"  ✓ PyAudio {pyaudio.__version__}")
    except ImportError:
        print("  ✗ PyAudio not installed")
    
    try:
        import flask
        print(f"  ✓ Flask {flask.__version__}")
    except ImportError:
        print("  ✗ Flask not installed")
    
    try:
        import psutil
        print(f"  ✓ psutil {psutil.__version__}")
    except ImportError:
        print("  ✗ psutil not installed")


if __name__ == '__main__':
    main()
