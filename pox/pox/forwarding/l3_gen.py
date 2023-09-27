from pox import core
from pox.lib.packet import packet
from pox.lib.packet import ethernet
from pox.lib.packet import ether_types
from pox.lib.packet import in_proto
from pox.lib.packet import ipv4
from pox.lib.packet import icmp
from pox.lib.packet import tcp
from pox.lib.packet import udp

class l3_gen(core.Component):
    def __init__(self, *args, **kwargs):
        super(l3_gen, self).__init__(*args, **kwargs)

        self.mac_to_port = {}

    def handle_PacketIn(self, event):
        packet = event.parsed

        # Ignore LLDP packets.
        if packet.type == ether_types.ETH_TYPE_LLDP:
            return

        src_mac = packet.src
        dst_mac = packet.dst
        in_port = event.port

        # Update the MAC-to-port table.
        self.mac_to_port[src_mac] = in_port

        # If the destination MAC address is in the MAC-to-port table,
        # send the packet out on the corresponding port.
        if dst_mac in self.mac_to_port:
            out_port = self.mac_to_port[dst_mac]
        else:
            out_port = core.openflow.OFPP_FLOOD

        # Construct an action list to send the packet out on the
        # specified port.
        actions = [core.openflow.ofp_action_output(port=out_port)]

        # If the packet is IP, add a flow rule to the switch so that
        # future packets with the same source and destination IP addresses
        # are forwarded directly to the destination port.
        if packet.type == ether_types.ETH_TYPE_IP:
            ip_packet = packet.find('ipv4')
            src_ip = ip_packet.srcip
            dst_ip = ip_packet.dstip

            match = core.openflow.ofp_match(
                dl_type=ether_types.ETH_TYPE_IP,
                nw_src=src_ip,
                nw_dst=dst_ip
            )

            flow = core.openflow.ofp_flow_mod()
            flow.match = match
            flow.actions = actions
            flow.priority = 1
            flow.idle_timeout = 20
            flow.hard_timeout = 100

            core.openflow.addListenerByName('PacketIn', flow)

        # Send the packet out on the specified port.
        msg = core.openflow.ofp_packet_out()
        msg.in_port = in_port
        msg.actions = actions
        msg.data = event.data
        core.openflow.sendToDP(event.dpid, msg)

def launch():
    core.registerNew(l3_gen)

if __name__ == '__main__':
    launch()
