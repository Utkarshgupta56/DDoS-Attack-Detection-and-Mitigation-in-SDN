# DDoS Attack Detection and Mitigation

## Overview
A robust, machine-learning-driven DDoS defense system for Software-Defined Networking (SDN), leveraging the Ryu SDN framework and Mininet. Simulates network traffic, detects attacks in real time with a Random Forest classifier, and dynamically mitigates threats using OpenFlow rules.

## Features
- **SDN-Based Architecture:** Centralized control with Ryu; flexible network simulation using Mininet.
- **Custom Topology:** 6 switches and 18 hosts, linear setup for realistic scenarios.
- **Traffic Simulation:** Benign (ICMP, TCP, UDP) and attack (TCP-SYN flood, etc.) flows.
- **Flow Collector:** Aggregates flow statistics (packet/byte counts, IPs, ports, protocols) for monitoring and ML.
- **DDoS Detection:** Trained Random Forest model; classifies network flows as benign or DDoS.
- **Real-Time Mitigation:** High-priority flow rules dynamically block detected attack traffic with 10s auto expiry.

## System Architecture

- **Network Topology:**  
  6 OpenFlow 1.3 switches (s1–s6), 18 hosts (h1–h18), interconnected in a linear topology.
- **Traffic Creation:**  
  Generate benign traffic (pings, HTTP downloads, iperf UDP/TCP) and DDoS attacks (TCP-SYN flood, configurable types).
- **Flow Collection:**  
  Periodic stats (every 10s for training, 5s during runtime) collected from all switches.
- **Detection & Prevention:**  
  ML pipeline trains the RF model and triggers flow rules to prevent ongoing attacks.

## Installation & Setup

1. **Make installer executable:**
 ```
  chmod +x install.sh
./install.sh
  ```

2. **Activate Python 3.7 environment:**
  ```
source ryu37-env/bin/activate
 ```
3. **Install required packages:**
```
pip3 install -r requirements.txt
```

4. **Run the system:**

- Start the Ryu SDN controller:
  ```
  ryu-manager Codes/controller/test_mitigate.py
  ```
- Start the traffic recorder:
  ```
  python3 recorder.py
  ```
- Run the Mininet traffic generation test:
  ```
  sudo python3 Codes/mininet/generate_ddos_trafic.py
  ```

## Results

- Accurately detects and mitigates DDoS attacks
- Maintains high accuracy with low false positives
- Temporary blocking ensures minimal disruption to legitimate users


