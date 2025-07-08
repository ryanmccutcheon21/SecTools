# SecTools
Explore a collection of cybersecurity tools, from penetration testing to bug bounty hunting. Empower your security assessments with a diverse set of utilities.

Tool Name: rbcd-auto.py

Description:
rbcd-auto.py is a Python-based exploitation tool designed to automate Resource-Based Constrained Delegation (RBCD) attacks against misconfigured Active Directory environments. It supports both:
  - Standard RBCD using machine account creation (when ms-DS-MachineAccountQuota > 0)
  - Forshaw's fallback method for environments where machine account creation is restricted (MAQ = 0), using a domain user's TGT + NT hash

The tool leverages the powerful Impacket toolkit and is intended for offensive security professionals, penetration testers, and red teamers.

Features
  - Automates full RBCD attack chain (computer creation → delegation → impersonation → SYSTEM shell)
  - Supports fallback to James Forshaw's technique when MAQ = 0
  - Works from Linux using Python 3 and Impacket
  - Dynamically sets KRB5CCNAME for Impacket tools
  - Auto-generates machine account name & password
  - Supports SPN impersonation targeting (CIFS, TERMSRV, LDAP, etc.)

Usage
  - Standard RBCD (MAQ > 0)
    
    python3 rbcd_auto.py \
      --dc-ip <DC_IP> \
      --dc-name <FQDN> \
      --domain <domain> \
      --username <username> \
      --password <password>
    
  - Forshaw Fallback (MAQ = 0, using owned user)

    python3 rbcd_auto.py \
      --dc-ip <DC_IP> \
      --dc-name <FQDN> \
      --domain <domain> \
      --username <username> \
      --password <password> \
      --target-spn <SPN> \
      --fallback

Dependencies
  - Python 3
  - Impacket (installed and in PATH)
  - Optional: xterm or gnome-terminal for interactive shell dropping

⚠️ Disclaimer
This tool is intended for educational purposes and authorized security assessments only. Do not use this script against systems you do not have explicit permission to test.
