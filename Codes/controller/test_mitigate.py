#!/usr/bin/env python3
"""
Ryu-based SDN application for DDoS detection and mitigation.
- Collects flow statistics periodically.
- Trains a RandomForest model at startup with a train/validation split.
- Predicts incoming traffic type and sets a mitigation flag.
- Blocks ports exhibiting DDoS traffic patterns.
- Logs output to both terminal and output.log file.
"""
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4, icmp, tcp, udp, in_proto

import logging
import sys



# Global counter for flow serial numbers
FLOW_SERIAL_NO = 0

def get_flow_number():
    global FLOW_SERIAL_NO
    FLOW_SERIAL_NO += 1
    return FLOW_SERIAL_NO

class MitigationSwitch(app_manager.RyuApp):
    """
    Combined SDN switch and DDoS mitigation application.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MitigationSwitch, self).__init__(*args, **kwargs)
        # Switch state
        self.mac_to_port = {}
        self.arp_ip_to_port = {}
        self.datapaths = {}
        # Mitigation flag (0: off, 1: on)
        self.mitigation = 0
        # ML model
        self.flow_model = None

        # Configure logger to output to both terminal and file
        self.logger = logging.getLogger('MitigationSwitch')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Terminal handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        # File handler
        file_handler = logging.FileHandler('output.log', mode='w')  # 'w' to overwrite on each run
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Start the monitoring thread
        self.monitor_thread = hub.spawn(self._monitor)

        # Train the model at startup
        start_time = datetime.now()
        self.flow_training()
        duration = datetime.now() - start_time
        self.logger.info(f"Model training time: {duration}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        # Install table-miss flow entry
        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        flow_serial = get_flow_number()
        self.add_flow(datapath, 0, match, actions, flow_serial)

    def add_flow(self, datapath, priority, match, actions,
                 serial_no, buffer_id=None, idle=0, hard=0):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath, cookie=serial_no, buffer_id=buffer_id,
                priority=priority, idle_timeout=idle, hard_timeout=hard,
                match=match, instructions=inst
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath, cookie=serial_no,
                priority=priority, idle_timeout=idle, hard_timeout=hard,
                match=match, instructions=inst
            )
        datapath.send_msg(mod)

    def block_port(self, datapath, port_no):
        """Block all incoming traffic on a specific port."""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch(in_port=port_no)
        actions = []  # drop
        flow_serial = get_flow_number()
        # High priority, hard timeout for auto-remove
        self.add_flow(datapath, 100, match, actions,
                      flow_serial, idle=0, hard=10)

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug(f"Register datapath: {datapath.id:016x}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug(f"Unregister datapath: {datapath.id:016x}")
                del self.datapaths[datapath.id]

    def _monitor(self):
        # Periodically request flow stats and predict
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)
            self.flow_predict()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        # Dump flow stats to CSV for prediction
        timestamp = datetime.now().timestamp()
        filename = 'PredictFlowStatsfile.csv'

        with open(filename, 'w') as f:
            f.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,'
                    'icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,'
                    'idle_timeout,hard_timeout,flags,packet_count,byte_count,'
                    'packet_count_per_second,packet_count_per_nsecond,'
                    'byte_count_per_second,byte_count_per_nsecond\n')

            body = ev.msg.body
            for stat in sorted([flow for flow in body if flow.priority == 1],
                               key=lambda f: (f.match.get('eth_type'),
                                              f.match.get('ipv4_src'),
                                              f.match.get('ipv4_dst'),
                                              f.match.get('ip_proto'))):
                ip_src = stat.match.get('ipv4_src', 0)
                ip_dst = stat.match.get('ipv4_dst', 0)
                ip_proto = stat.match.get('ip_proto', 0)
                icmp_code = -1
                icmp_type = -1
                tp_src = 0
                tp_dst = 0

                if ip_proto == in_proto.IPPROTO_ICMP:
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)
                elif ip_proto == in_proto.IPPROTO_TCP:
                    tp_src = stat.match.get('tcp_src', 0)
                    tp_dst = stat.match.get('tcp_dst', 0)
                elif ip_proto == in_proto.IPPROTO_UDP:
                    tp_src = stat.match.get('udp_src', 0)
                    tp_dst = stat.match.get('udp_dst', 0)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                # Compute rates safely
                def safe_div(a, b): return a/b if b else 0
                pps = safe_div(stat.packet_count, stat.duration_sec)
                ppns = safe_div(stat.packet_count, stat.duration_nsec)
                bps = safe_div(stat.byte_count, stat.duration_sec)
                bpns = safe_div(stat.byte_count, stat.duration_nsec)

                f.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},"
                        f"{ip_src},{tp_src},{ip_dst},{tp_dst},"
                        f"{ip_proto},{icmp_code},{icmp_type},"
                        f"{stat.duration_sec},{stat.duration_nsec},"
                        f"{stat.idle_timeout},{stat.hard_timeout},"
                        f"{stat.flags},{stat.packet_count},{stat.byte_count},"
                        f"{pps},{ppns},{bps},{bpns}\n")

    def flow_training(self):
        """Train RandomForest with train/validation split."""
        self.logger.info("Flow Training ...")
        df = pd.read_csv('FlowStatsfile.csv')
        # Clean numeric columns stored as strings
        for col_idx in [2, 3, 5]:
            df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(str).str.replace('.', '',regex=False)

        df = df.drop(df.columns[0], axis=1)  # Drop the first column (timestamp)
        X = df.iloc[:, :-1].astype('float64').values
        y = df.iloc[:, -1].values

        # 60% train, 20% val, 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=0)

        clf = RandomForestClassifier(
            n_estimators=100, criterion='gini', random_state=0)
        self.flow_model = clf.fit(X_train, y_train)

        # Log validation accuracy
        y_val_pred = self.flow_model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        self.logger.info(f"Validation Accuracy: {val_acc:.4f}")

    def flow_predict(self):
        """Read the dumped CSV, predict, and set mitigation flag."""
        try:
            df = pd.read_csv('PredictFlowStatsfile.csv')
            for col_idx in [2, 3, 5]:
                df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(str).str.replace('.', '', regex=False)

            df = df.drop(df.columns[0], axis=1)  # Drop the first column (timestamp)
            Xp = df.astype('float64').values

            preds = self.flow_model.predict(Xp)
            legit = (preds == 0).sum()
            ddos = (preds == 1).sum()
            ratio = legit / len(preds)

            self.logger.info("-"*60)
            if ratio > 0.8:
                self.logger.info("Traffic is Legitimate!")
                self.mitigation = 0
            else:
                self.logger.info("NOTICE!! DoS Attack in Progress!!!")
                self.mitigation = 1

            # Prepare for next cycle
            open('PredictFlowStatsfile.csv', 'w').write(
                'timestamp,datapath_id,flow_id,ip_src,...\n')
        except Exception:
            pass

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Normal learning-switch behavior, with blocking when mitigation is on."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})

        # Learn MAC
        self.mac_to_port[dpid][eth.src] = in_port

        # Decide output port
        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        # Mitigation: drop packet and block port if attack detected
        if self.mitigation:
            self.logger.info(
                f"[MITIGATION] Blocking port {in_port} on switch {dpid}")
            self.block_port(datapath, in_port)
            return

        # Install a flow to avoid future packet_in
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            proto = ip_pkt.proto
            match_kwargs = dict(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst,
                ip_proto=proto
            )
            if proto == in_proto.IPPROTO_ICMP:
                icmppkt = pkt.get_protocol(icmp.icmp)
                match_kwargs.update(
                    icmpv4_code=icmppkt.code,
                    icmpv4_type=icmppkt.type
                )
            elif proto == in_proto.IPPROTO_TCP:
                tcppkt = pkt.get_protocol(tcp.tcp)
                match_kwargs.update(
                    tcp_src=tcppkt.src_port,
                    tcp_dst=tcppkt.dst_port
                )
            elif proto == in_proto.IPPROTO_UDP:
                udppkt = pkt.get_protocol(udp.udp)
                match_kwargs.update(
                    udp_src=udppkt.src_port,
                    udp_dst=udppkt.dst_port
                )
            match = parser.OFPMatch(**match_kwargs)
            flow_serial = get_flow_number()
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(
                    datapath, 1, match, actions,
                    flow_serial, buffer_id=msg.buffer_id,
                    idle=20, hard=100
                )
                return
            else:
                self.add_flow(
                    datapath, 1, match, actions,
                    flow_serial, idle=20, hard=100
                )

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)