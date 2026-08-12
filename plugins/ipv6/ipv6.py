"""Example /ipv6 plugin."""

from xmagents.plugins import local_ipv6_addresses

PLUGIN = {
    "name": "IPv6",
    "command": "ipv6",
    "description": "返回设备 IPv6 地址",
    "usage": "/ipv6",
}


def handle(ctx, args: str) -> str:
    addresses = local_ipv6_addresses()
    return "\n".join(addresses) if addresses else "未发现非回环 IPv6 地址。"
