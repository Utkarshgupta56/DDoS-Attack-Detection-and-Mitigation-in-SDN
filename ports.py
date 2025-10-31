import re
from collections import defaultdict

# Initialize a dictionary to hold sets of blocked ports for each switch
blocked_ports = defaultdict(set)

# Read the log file
with open("output.log", "r") as file:  # Replace with your actual log filename
    for line in file:
        match = re.search(r"Blocking port (\d+) on switch (\d+)", line)
        if match:
            port = int(match.group(1))
            switch = int(match.group(2))
            blocked_ports[switch].add(port)

# Print the result in the required format
for switch in sorted(blocked_ports.keys()):
    ports = sorted(blocked_ports[switch])
    ports_str = ", ".join(str(port) for port in ports)
    print(f"s{switch}: {ports_str}")
