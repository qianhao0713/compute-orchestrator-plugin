#!/bin/bash
# 获取BKCL_SOCKET_IFNAME & GLOO_SOCKET_IFNAME
get_real_iface() {
    local target=${1:-8.8.8.8}
    local dev=""

    if [ -n "$target" ]; then
        dev=$(ip route get "$target" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
    fi

    # 如果上面失败，尝试从 default 路由取
    if [ -z "$dev" ]; then
        dev=$(ip route | awk '/^default/{print $5; exit}')
    fi

    # 如果还是没有 default 路由，取第一条非 blackhole 路由的出口设备
    if [ -z "$dev" ]; then
        dev=$(ip route | awk '!/^blackhole/ && /dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
    fi

    # 递归解析隧道/虚拟接口（原逻辑不变）
    while true; do
        case "$dev" in
            natgre*|gre*|gretap*|erspan*|tun*|vxlan* )
                peer=$(ip -d link show "$dev" 2>/dev/null | grep -oP 'remote \K[0-9.]+' | head -n1)
                [ -z "$peer" ] && break
                dev=$(ip route get "$peer" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
                ;;
            veth*|docker*|virbr* )
                dev=$(ip route | awk '/^default/{print $5; exit}')
                [ -z "$dev" ] && dev=$(ip route | awk '!/^blackhole/ && /dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
                ;;
            *)
                break
                ;;
        esac
    done

    echo "$dev"
}

get_real_iface