"""
Dependency Checker for Thoth Device

Automatically checks and installs missing dependencies on startup.
"""

import subprocess
import sys
import platform
import os
import logging

logger = logging.getLogger(__name__)


def check_and_install_dependencies():
    """Check for required dependencies and install if missing.
    
    Called automatically on app startup.
    """
    missing = []
    
    # Check OpenCV
    try:
        import cv2
        logger.info(f"OpenCV {cv2.__version__} available")
    except ImportError:
        missing.append(('opencv-python', 'cv2'))
    
    # Check PyAudio (platform-specific)
    try:
        import pyaudio
        logger.info(f"PyAudio {pyaudio.__version__} available")
    except ImportError:
        missing.append(('pyaudio', 'pyaudio'))
    
    # Check psutil
    try:
        import psutil
        logger.info(f"psutil {psutil.__version__} available")
    except ImportError:
        missing.append(('psutil', 'psutil'))
    
    # Check netifaces
    try:
        import netifaces
        logger.info("netifaces available")
    except ImportError:
        missing.append(('netifaces', 'netifaces'))
    
    if not missing:
        logger.info("All dependencies are installed")
        return True
    
    logger.warning(f"Missing dependencies: {[m[0] for m in missing]}")
    
    # Try to install missing dependencies
    for package, module in missing:
        if package == 'pyaudio':
            success = _install_pyaudio()
        else:
            success = _pip_install(package)
        
        if success:
            logger.info(f"Successfully installed {package}")
        else:
            logger.warning(f"Failed to install {package} - some features may not work")
    
    return True


def _pip_install(package: str) -> bool:
    """Install a package using pip."""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package, '--quiet'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _install_pyaudio() -> bool:
    """Install PyAudio with platform-specific handling."""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        return _install_pyaudio_macos()
    elif system == 'Linux':
        return _install_pyaudio_linux()
    elif system == 'Windows':
        return _pip_install('pyaudio')
    else:
        return _pip_install('pyaudio')


def _install_pyaudio_macos() -> bool:
    """Install PyAudio on macOS."""
    # Check if Homebrew is available
    try:
        subprocess.run(['brew', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Homebrew not installed. Cannot auto-install PyAudio on macOS.")
        logger.warning("Run: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        return False
    
    # Install portaudio
    try:
        logger.info("Installing portaudio via Homebrew...")
        subprocess.run(['brew', 'install', 'portaudio'], capture_output=True, check=False)
    except Exception as e:
        logger.warning(f"Failed to install portaudio: {e}")
    
    # Determine Homebrew prefix
    try:
        result = subprocess.run(['brew', '--prefix'], capture_output=True, text=True, check=True)
        brew_prefix = result.stdout.strip()
    except:
        brew_prefix = '/opt/homebrew' if platform.machine() == 'arm64' else '/usr/local'
    
    # Install PyAudio with correct flags
    env = os.environ.copy()
    env['CFLAGS'] = f"-I{brew_prefix}/include -L{brew_prefix}/lib"
    env['LDFLAGS'] = f"-L{brew_prefix}/lib"
    
    try:
        logger.info("Installing PyAudio...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'pyaudio'],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        logger.warning("Failed to install PyAudio. Microphone features will be unavailable.")
        return False


def _install_pyaudio_linux() -> bool:
    """Install PyAudio on Linux."""
    # Try to install system dependencies first
    distro_cmds = [
        ['sudo', 'apt-get', 'install', '-y', 'python3-pyaudio', 'portaudio19-dev'],
        ['sudo', 'dnf', 'install', '-y', 'python3-pyaudio', 'portaudio-devel'],
        ['sudo', 'pacman', '-S', '--noconfirm', 'python-pyaudio', 'portaudio'],
    ]
    
    for cmd in distro_cmds:
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=60)
            return True
        except:
            continue
    
    # Fallback to pip
    return _pip_install('pyaudio')


def ensure_data_directory():
    """Ensure the data directory exists with proper structure."""
    from backend.config import Config
    
    directories = [
        Config.DATA_DIR,
        os.path.join(Config.DATA_DIR, 'config'),
        Config.LOGS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    logger.info(f"Data directory ready: {Config.DATA_DIR}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    check_and_install_dependencies()
    ensure_data_directory()
