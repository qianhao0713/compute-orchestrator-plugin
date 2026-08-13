#!/bin/bash
# 获取 XPU 卡对应的 RDMA 网卡配置脚本（本地版本）
# 用法: ./get_rdma_config.sh
# 输出格式: XPU0对应的网卡,XPU1对应的网卡,...

# 主函数
main() {
    # 直接执行 xpu-smi topo -m
    local topo_output=$(xpu-smi topo -m 2>/dev/null)

    if [[ -z "$topo_output" ]]; then
        echo "错误: xpu-smi topo -m 执行失败"
        return 1
    fi

    # 解析 NIC Legend
    declare -A nic_to_phys
    local in_legend=0
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*NIC([0-9]+):[[:space:]]+([a-z0-9_]+) ]]; then
            nic_to_phys[${BASH_REMATCH[1]}]=${BASH_REMATCH[2]}
        fi
    done <<< "$topo_output"

    # 使用 sudo 获取 ibdev2netdev 的所有输出
    local ibdev_output=$(sudo ibdev2netdev 2>/dev/null)

    # 解析每个 XPU 对应的网卡
    local result=""
    for i in {0..7}; do
        local xpu_line=$(echo "$topo_output" | grep "^XPU$i")
        
        # 去除 ANSI 转义码
        xpu_line=$(echo "$xpu_line" | sed 's/\x1b\[[0-9;]*m//g')
        
        # 分割字段
        fields=($(echo "$xpu_line" | tr -s '[:space:]' '\n'))
        
        # 字段 9-18 对应 NIC0-NIC9
        # 收集所有 PIX 的 NIC
        local pix_nics=()
        for j in {0..9}; do
            local idx=$((j + 9))
            if [[ -n "${fields[$idx]}" && "${fields[$idx]}" == "PIX" ]]; then
                pix_nics+=("${nic_to_phys[$j]}")
            fi
        done

        # 如果有多个NIC，优先选择状态为Up的，没有Up再选择Down的
        local found_nic=""
        if [[ ${#pix_nics[@]} -gt 0 ]]; then
            # 先找状态为Up的NIC
            for nic in "${pix_nics[@]}"; do
                local status=$(echo "$ibdev_output" | grep -w "$nic" | awk '{print $6}')
                if [[ "$status" == "(Up)" ]]; then
                    found_nic="$nic"
                    break
                fi
            done
            # 如果没有Up的，选择第一个（通常是Down）
            if [[ -z "$found_nic" ]]; then
                found_nic="${pix_nics[0]}"
            fi
        fi

        # 获取逻辑网卡名
        local log_nic
        if [[ -n "$found_nic" ]]; then
            log_nic=$(echo "$ibdev_output" | grep -w "$found_nic" | awk '{print $5}')
            [[ -z "$log_nic" ]] && log_nic="$found_nic"
        else
            log_nic="N/A"
        fi

        if [[ -z "$result" ]]; then
            result="$log_nic"
        else
            result="$result,$log_nic"
        fi
    done

    echo "$result"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi