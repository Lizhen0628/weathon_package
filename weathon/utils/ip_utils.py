# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 11:19
# @Author  : LiZhen
# @FileName: ip_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import paramiko
from tqdm import tqdm
from typing import Set


class IpUtils:

    @staticmethod
    def find_server_ipv4(cls, username, password,
                         ip1: int = None, ip2: int = None, ip3: int = None, ip4: int = None,port: int = 22) -> Set:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        all_ip = set()
        wrong_ip = set()

        ip1_range = [ip1] if ip1 else list(range(256))
        ip2_range = [ip2] if ip2 else list(range(256))
        ip3_range = [ip3] if ip3 else list(range(256))
        ip4_range = [ip4] if ip4 else list(range(256))

        for i in ip1_range:
            for j in ip2_range:
                for x in ip3_range:
                    for y in ip4_range:
                        ip = f"{i}.{j}.{x}.{y}"
                        all_ip.add(ip)
                        print(ip)
                        try:
                            ssh.connect(hostname=ip, port=port, username=username, password=password, timeout=0.5)
                        except Exception:
                            wrong_ip.add(ip)

        return set(all_ip).difference(set(wrong_ip))


if __name__ == '__main__':
    ip_utils = IpUtils()
    print(ip_utils.find_server_ipv4('lizhen','lizhen123',192,168,1))