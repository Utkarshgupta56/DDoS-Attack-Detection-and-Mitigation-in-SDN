#!/usr/bin/env python3

import subprocess
import time
import os
import re

# List of switches to monitor
switches = ["s1", "s2", "s3", "s4", "s5", "s6"]
# Output file to store blocked ports
output_file = "blocked_ports_per_switch.txt"

# Dictionary to keep track of blocked ports per switch
blocked_ports = {sw: set() for sw in switches}

def extract_blocked_port(flow_line):
    """
    Extracts the port from a flow line with a drop action and priority 100
    """
    if "priority=100" in flow_line and "actions=drop" in flow_line:
        match = re.search(r'in_port="?([\w\-]+)"?', flow_line)
        if match:
            return match.group(1)
    return None

while True:
    for sw in switches:
        try:
            result = subprocess.run(
                ["sudo", "ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                port = extract_blocked_port(line)
                if port:
                    blocked_ports[sw].add(port)

        except subprocess.CalledProcessError as e:
            print(f"Error querying {sw}: {e}")

    # Rewrite the output file with the latest blocked ports
    with open(output_file, "w") as f:
        for sw, ports in blocked_ports.items():
            if ports:
                port_list = ", ".join(sorted(ports))
                f.write(f"{sw}: {port_list}\n")
            else:
                f.write(f"{sw}: None\n")

    time.sleep(5)
