#!/usr/bin/env python3
import argparse
import hashlib
import time
import gzip
import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Try to import colorama for colored output
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS = {
        "info": Fore.CYAN,
        "success": Fore.GREEN,
        "error": Fore.RED,
        "bold": Style.BRIGHT,
        "dim": Style.DIM,
        "reset": Style.RESET_ALL,
    }
except ImportError:
    COLORS = {k: "" for k in ["info", "success", "error", "bold", "dim", "reset"]}


def color(text, color_key, enable=True):
    if enable:
        return COLORS.get(color_key, "") + text + COLORS.get("reset", "")
    return text


# Rule-based mangling (basic)
def mangle(word):
    rules = [
        lambda w: w,
        lambda w: w.capitalize(),
        lambda w: w.upper(),
        lambda w: w + "123",
        lambda w: w + "1",
        lambda w: w[::-1],
        lambda w: w + "!",
    ]
    return [rule(word) for rule in rules]


def parse_full_hash(full_hash):
    parts = full_hash.split("$")
    if len(parts) != 3:
        raise ValueError(f"Invalid PBKDF2 hash format. Got {len(parts)} parts, expected 3.")
    prefix, salt, digest = parts
    try:
        scheme, alg, iterations = prefix.split(":")
        return alg, int(iterations), salt, digest
    except Exception as e:
        raise ValueError(f"Invalid PBKDF2 prefix format: {e}")


def check_password(args):
    password, salt, iterations, target_hash, alg = args
    try:
        computed = hashlib.pbkdf2_hmac(
            alg, password.encode(), salt.encode(), iterations
        ).hex()
        if computed == target_hash:
            return password
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="PBKDF2 Cracker (rules, multiprocessing, progress, colors)")
    parser.add_argument("fullhash", nargs="?", help="Full hash (e.g. pbkdf2:sha256:600000$salt$hash)")
    parser.add_argument("-s", "--salt", help="Salt value")
    parser.add_argument("-H", "--hash", dest="target_hash", help="Target hash (hex)")
    parser.add_argument("-i", "--iterations", type=int, help="Iteration count")
    parser.add_argument("-a", "--algorithm", default="sha256", choices=["sha1", "sha256", "sha512"], help="Hash algorithm")
    parser.add_argument("-w", "--wordlist", default="rockyou.txt.gz", help="Wordlist file (txt or gz)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    use_color = not args.no_color
    verbosity = 1
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0

    if not args.fullhash and not (args.salt and args.target_hash and args.iterations):
        parser.print_help()
        return

    # Parse hash
    if args.fullhash:
        try:
            alg, iterations, salt, target_hash = parse_full_hash(args.fullhash)
        except ValueError as e:
            print(color(f"[!] {e}", "error", use_color))
            return
    else:
        alg = args.algorithm
        iterations = args.iterations
        salt = args.salt
        target_hash = args.target_hash

    if verbosity >= 1:
        print(color(f"[+] Algorithm     : {alg}", "info", use_color))
        print(color(f"[+] Iterations    : {iterations}", "info", use_color))
        print(color(f"[+] Salt          : {salt}", "info", use_color))
        print(color(f"[+] Target Hash   : {target_hash}", "info", use_color))
        print(color(f"[+] Wordlist      : {args.wordlist}", "info", use_color))

    if not os.path.exists(args.wordlist):
        print(color(f"[-] Wordlist file not found: {args.wordlist}", "error", use_color))
        return

    # Load wordlist
    try:
        if args.wordlist.endswith(".gz"):
            with gzip.open(args.wordlist, "rt", errors="ignore") as f:
                base_words = f.read().splitlines()
        else:
            with open(args.wordlist, "r", errors="ignore") as f:
                base_words = f.read().splitlines()
    except Exception as e:
        print(color(f"[-] Error reading wordlist: {e}", "error", use_color))
        return

    # Apply basic mangling
    candidates = []
    for word in base_words:
        candidates.extend(mangle(word.strip()))

    if verbosity >= 1:
        print(color(f"[+] Candidates after mangling: {len(candidates)}", "info", use_color))

    pool_args = [(word, salt, iterations, target_hash, alg) for word in candidates]

    found = None
    start = time.time()
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(check_password, pool_args),
                           total=len(pool_args),
                           desc="Cracking",
                           unit="hash",
                           disable=verbosity == 0):
            if result:
                found = result
                pool.terminate()
                break

    duration = time.time() - start
    speed = len(pool_args) / duration if duration else 0

    if verbosity >= 1:
        print(color(f"[+] Time: {duration:.2f}s | Speed: {speed:.2f} H/s", "info", use_color))

    if found:
        print(color(f"[✔] Password found: {found}", "success", use_color))
    else:
        print(color("[-] Password not found.", "error", use_color))


if __name__ == "__main__":
    main()
