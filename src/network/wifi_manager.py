"""WiFi and Access Point management module.

This module handles automatic WiFi connection, AP mode switching,
and captive portal functionality for WiFi configuration.
"""

import os
import sys
import subprocess
import time
import logging
import argparse
from typing import Optional, Dict, Any

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WiFiManager:
    """Manages WiFi connections and AP mode."""
    
    def __init__(self):
        self.ap_ssid = Config.AP_SSID
        self.ap_password = Config.AP_PASSWORD
        self.interface = "wlan0"
        self.ap_ip = "192.168.4.1"
        self.ap_range = "192.168.4.2,192.168.4.20"
        
        # File paths
        self.hostapd_conf = "/tmp/thoth_hostapd.conf"
        self.dnsmasq_conf = "/tmp/thoth_dnsmasq.conf"
        self.wpa_supplicant_conf = "/etc/wpa_supplicant/wpa_supplicant.conf"
        
    def is_connected(self) -> bool:
        """Check if connected to internet."""
        try:
            result = subprocess.run(
                ["ping", "-c1", "-W3", "8.8.8.8"], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            return False
    
    def get_wifi_status(self) -> Dict[str, Any]:
        """Get current WiFi status."""
        try:
            # Check interface status
            iwconfig_result = subprocess.run(
                ["iwconfig", self.interface],
                capture_output=True, text=True
            )
            
            # Check if connected to a network
            connected_ssid = None
            if "ESSID:" in iwconfig_result.stdout:
                essid_line = [line for line in iwconfig_result.stdout.split('\n') if 'ESSID:' in line]
                if essid_line:
                    connected_ssid = essid_line[0].split('ESSID:')[1].strip().strip('"')
            
            # Get IP address
            ip_result = subprocess.run(
                ["ip", "addr", "show", self.interface],
                capture_output=True, text=True
            )
            
            ip_address = None
            for line in ip_result.stdout.split('\n'):
                if 'inet ' in line and not '127.0.0.1' in line:
                    ip_address = line.strip().split()[1].split('/')[0]
                    break
            
            return {
                "connected": self.is_connected(),
                "ssid": connected_ssid,
                "ip_address": ip_address,
                "interface_up": "UP" in iwconfig_result.stdout
            }
        except Exception as e:
            logger.error(f"Error getting WiFi status: {e}")
            return {"connected": False, "error": str(e)}
    
    def scan_networks(self) -> list:
        """Scan for available WiFi networks."""
        try:
            # Trigger scan
            subprocess.run(["sudo", "iwlist", self.interface, "scan"], capture_output=True)
            time.sleep(2)
            
            # Get scan results
            result = subprocess.run(
                ["sudo", "iwlist", self.interface, "scan"],
                capture_output=True, text=True
            )
            
            networks = []
            current_network = {}
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if 'Cell ' in line and 'Address:' in line:
                    if current_network:
                        networks.append(current_network)
                    current_network = {"address": line.split('Address: ')[1]}
                elif 'ESSID:' in line:
                    essid = line.split('ESSID:')[1].strip().strip('"')
                    if essid:
                        current_network["ssid"] = essid
                elif 'Quality=' in line:
                    quality = line.split('Quality=')[1].split()[0]
                    current_network["quality"] = quality
                elif 'Encryption key:' in line:
                    encrypted = 'on' in line
                    current_network["encrypted"] = encrypted
            
            if current_network:
                networks.append(current_network)
            
            return [n for n in networks if 'ssid' in n]
            
        except Exception as e:
            logger.error(f"Error scanning networks: {e}")
            return []
    
    def connect_wifi(self, ssid: str, password: str = None) -> bool:
        """Connect to a WiFi network."""
        try:
            logger.info(f"Attempting to connect to WiFi: {ssid}")
            
            # Create network configuration
            network_config = f'''
network={{
    ssid="{ssid}"
    {f'psk="{password}"' if password else 'key_mgmt=NONE'}
    priority=1
}}
'''
            
            # Add to wpa_supplicant configuration
            with open(self.wpa_supplicant_conf, 'a') as f:
                f.write(network_config)
            
            # Reconfigure wpa_supplicant
            subprocess.run(["sudo", "wpa_cli", "-i", self.interface, "reconfigure"])
            
            # Wait for connection
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.is_connected():
                    logger.info(f"Successfully connected to {ssid}")
                    return True
            
            logger.warning(f"Failed to connect to {ssid} within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to WiFi: {e}")
            return False
    
    def create_hostapd_config(self):
        """Create hostapd configuration file."""
        config = f"""
interface={self.interface}
driver=nl80211
ssid={self.ap_ssid}
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase={self.ap_password}
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
"""
        with open(self.hostapd_conf, 'w') as f:
            f.write(config)
    
    def create_dnsmasq_config(self):
        """Create dnsmasq configuration file."""
        config = f"""
interface={self.interface}
dhcp-range={self.ap_range},255.255.255.0,24h
dhcp-option=3,{self.ap_ip}
dhcp-option=6,{self.ap_ip}
server=8.8.8.8
log-queries
log-dhcp
listen-address={self.ap_ip}
"""
        with open(self.dnsmasq_conf, 'w') as f:
            f.write(config)
    
    def setup_ap_interface(self):
        """Configure network interface for AP mode."""
        try:
            # Stop any existing network services
            subprocess.run(["sudo", "systemctl", "stop", "dhcpcd"], capture_output=True)
            subprocess.run(["sudo", "systemctl", "stop", "wpa_supplicant"], capture_output=True)
            
            # Configure interface
            subprocess.run(["sudo", "ip", "link", "set", self.interface, "down"])
            subprocess.run(["sudo", "ip", "addr", "flush", "dev", self.interface])
            subprocess.run(["sudo", "ip", "addr", "add", f"{self.ap_ip}/24", "dev", self.interface])
            subprocess.run(["sudo", "ip", "link", "set", self.interface, "up"])
            
            logger.info(f"Interface {self.interface} configured for AP mode")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up AP interface: {e}")
            return False
    
    def start_ap(self) -> bool:
        """Start Access Point mode."""
        try:
            logger.info("Starting Access Point mode")
            
            # Setup interface
            if not self.setup_ap_interface():
                return False
            
            # Create configuration files
            self.create_hostapd_config()
            self.create_dnsmasq_config()
            
            # Start hostapd
            hostapd_process = subprocess.Popen([
                "sudo", "hostapd", self.hostapd_conf
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(3)  # Give hostapd time to start
            
            # Check if hostapd is running
            if hostapd_process.poll() is not None:
                stdout, stderr = hostapd_process.communicate()
                logger.error(f"hostapd failed to start: {stderr.decode()}")
                return False
            
            # Start dnsmasq
            dnsmasq_result = subprocess.run([
                "sudo", "dnsmasq", "-C", self.dnsmasq_conf, "-d"
            ], capture_output=True)
            
            if dnsmasq_result.returncode != 0:
                logger.error(f"dnsmasq failed to start: {dnsmasq_result.stderr.decode()}")
                hostapd_process.terminate()
                return False
            
            # Setup iptables for internet sharing (if ethernet available)
            try:
                subprocess.run([
                    "sudo", "iptables", "-t", "nat", "-A", "POSTROUTING", 
                    "-o", "eth0", "-j", "MASQUERADE"
                ], capture_output=True)
                subprocess.run([
                    "sudo", "iptables", "-A", "FORWARD", "-i", "eth0", 
                    "-o", self.interface, "-m", "state", "--state", "RELATED,ESTABLISHED", "-j", "ACCEPT"
                ], capture_output=True)
                subprocess.run([
                    "sudo", "iptables", "-A", "FORWARD", "-i", self.interface, 
                    "-o", "eth0", "-j", "ACCEPT"
                ], capture_output=True)
                subprocess.run(["sudo", "sysctl", "net.ipv4.ip_forward=1"], capture_output=True)
            except Exception as e:
                logger.warning(f"Could not setup internet sharing: {e}")
            
            logger.info(f"Access Point started: SSID={self.ap_ssid}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting AP: {e}")
            return False
    
    def stop_ap(self):
        """Stop Access Point mode."""
        try:
            logger.info("Stopping Access Point mode")
            
            # Stop services
            subprocess.run(["sudo", "pkill", "hostapd"], capture_output=True)
            subprocess.run(["sudo", "pkill", "dnsmasq"], capture_output=True)
            
            # Clean up iptables
            subprocess.run(["sudo", "iptables", "-F"], capture_output=True)
            subprocess.run(["sudo", "iptables", "-t", "nat", "-F"], capture_output=True)
            
            # Restart normal networking
            subprocess.run(["sudo", "systemctl", "start", "dhcpcd"], capture_output=True)
            subprocess.run(["sudo", "systemctl", "start", "wpa_supplicant"], capture_output=True)
            
            logger.info("Access Point stopped")
            
        except Exception as e:
            logger.error(f"Error stopping AP: {e}")
    
    def auto_manage(self):
        """Automatically manage WiFi connection and AP mode."""
        logger.info("Starting WiFi auto-management")
        
        while True:
            try:
                if self.is_connected():
                    logger.debug("WiFi connected, checking again in 30 seconds")
                    time.sleep(30)
                else:
                    logger.info("No WiFi connection, starting AP mode")
                    self.start_ap()
                    
                    # Wait in AP mode for 5 minutes, then try WiFi again
                    time.sleep(300)
                    self.stop_ap()
                    time.sleep(10)  # Brief pause before checking WiFi again
                    
            except KeyboardInterrupt:
                logger.info("WiFi manager stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in auto-management: {e}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    """Main function for running as a service or standalone."""
    parser = argparse.ArgumentParser(description='Thoth WiFi Manager')
    parser.add_argument('--start-ap', action='store_true', help='Start AP mode')
    parser.add_argument('--stop-ap', action='store_true', help='Stop AP mode')
    parser.add_argument('--status', action='store_true', help='Show WiFi status')
    parser.add_argument('--scan', action='store_true', help='Scan for networks')
    parser.add_argument('--auto', action='store_true', help='Auto-manage WiFi/AP')
    
    args = parser.parse_args()
    
    wifi_manager = WiFiManager()
    
    if args.start_ap:
        success = wifi_manager.start_ap()
        print(f"AP start: {'Success' if success else 'Failed'}")
    elif args.stop_ap:
        wifi_manager.stop_ap()
        print("AP stopped")
    elif args.status:
        status = wifi_manager.get_wifi_status()
        print(f"WiFi Status: {status}")
    elif args.scan:
        networks = wifi_manager.scan_networks()
        print(f"Found {len(networks)} networks:")
        for net in networks:
            print(f"  {net.get('ssid', 'Unknown')} - {net.get('quality', 'Unknown')} - {'Encrypted' if net.get('encrypted', False) else 'Open'}")
    elif args.auto:
        wifi_manager.auto_manage()
    else:
        # Default: auto-manage
        wifi_manager.auto_manage()

if __name__ == '__main__':
    main()
