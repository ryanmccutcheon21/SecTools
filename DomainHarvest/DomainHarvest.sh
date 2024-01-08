#!/bin/bash

# DomainHarvest
# Author: Ryan McCutcheon
# Version: 1.0
# Date: January 10, 2024

usage() {
  echo "Usage: $0 [-t <target>] [-l] [-c] [-h] [-f] [-w <wordlist>] [-s <subdomains_file>] [-r] [-rd <depth>]"
  echo "Options:"
  echo "  -t <target>        Set the target domain."
  echo "  -l                Show all subdomains."
  echo "  -c                Show the count of domains found."
  echo "  -h                Show this help message with tool introduction."
  echo "  -f                Enable fuzzing of subdomains."
  echo "  -w <wordlist>     Specify the path to the wordlist for fuzzing. (Required with -f)"
  echo "  -s <subdomains_file> Specify the path to an existing subdomains file for post-processing. (Required without -t)"
  echo "  -r                Enable recursive fuzzing."
  echo "  -rd <depth>        Set the depth for recursive fuzzing."
  exit 1
}

tool_intro() {
  echo "====================================================================="
  echo "====================================================================="
  echo "Custom Domain Enumeration Tool"
  echo "This tool automates the process of collecting subdomains using various sources."
  echo "It supports multiple functionalities such as running theHarvester, extracting, and sorting subdomains."
  echo "It also provides options for fuzzing subdomains with ffuf."
  echo "====================================================================="
  echo "Example: DomainHarvest -t <target> -l -c -f -w <wordlist> -s <subdomains_file> -r -rd <depth>"
  echo "====================================================================="
  echo "Options:"
  echo "  -t <target>        Set the target domain."
  echo "  -l                Show all subdomains."
  echo "  -c                Show the count of domains found."
  echo "  -h                Show this help message with tool introduction."
  echo "  -f                Enable fuzzing of subdomains."
  echo "  -w <wordlist>     Specify the path to the wordlist for fuzzing. (Required with -f)"
  echo "  -s <subdomains_file> Specify the path to an existing subdomains file for post-processing. (Required without -t)"
  echo "  -r                Enable recursive fuzzing."
  echo "  -rd <depth>        Set the depth for recursive fuzzing."
  echo "====================================================================="
  echo "====================================================================="
  exit 0
}

# Initialize variables
SHOW_SUBDOMAINS=false
SHOW_COUNT=false
FUZZ_SUBDOMAINS=false
RECURSIVE_FUZZ=false
FUZZ_DEPTH=1
WORDLIST=""
SUBDOMAINS_FILE=""
TARGET=""
TARGET_SET=false
WORDLIST_SET=false
SUBDOMAINS_SET=false

# Parse command line options
while getopts ":t:lc:hfw:s:rd:" opt; do
  case $opt in
    t)
      TARGET="$OPTARG"
      TARGET_SET=true
      ;;
    l)
      SHOW_SUBDOMAINS=true
      ;;
    c)
      SHOW_COUNT=true
      ;;
    h)
      tool_intro
      ;;
    f)
      FUZZ_SUBDOMAINS=true
      ;;
    w)
      WORDLIST="$OPTARG"
      WORDLIST_SET=true
      ;;
    s)
      SUBDOMAINS_FILE="$OPTARG"
      SUBDOMAINS_SET=true
      ;;
    r)
      RECURSIVE_FUZZ=true
      ;;
    rd)
      FUZZ_DEPTH="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done

# Step 1: Create sources.txt
if [ "$TARGET_SET" = true ]; then
  touch sources.txt
  mkdir -p files

  # Add sources to sources.txt
  echo "baidu
  bufferoverun
  crtsh
  hackertarget
  otx
  projectdiscovery
  rapiddns
  sublist3r
  threatcrowd
  trello
  urlscan
  vhost 
  virustotal
  zoomeye" > sources.txt

  # Step 3: Run theHarvester
  cat sources.txt | while read source; do
    theHarvester -d "${TARGET}" -b $source -f "./files/${source}_${TARGET}"
  done

  # Step 4: Extract and sort subdomains
  cat ./files/*.json 2>/dev/null | jq -r '.hosts[]' | cut -d':' -f 1 | sort -u > "./files/${TARGET}_theHarvester.txt"

  # Step 5: Merge files
  cat "./files/${TARGET}_theHarvester.txt" | sort -u > "${TARGET}_subdomains_passive.txt"

  # Step 6 (Optional): Show all subdomains
  if [ "$SHOW_SUBDOMAINS" = true ]; then
    cat "${TARGET}_subdomains_passive.txt"
  fi

  # Step 7 (Optional): Show the count of domains found
  if [ "$SHOW_COUNT" = true ]; then
    wc -l < "${TARGET}_subdomains_passive.txt"
  fi

# If -t is not specified, assume fuzzing is intended
elif [ -z "$TARGET" ]; then
  if [ "$SUBDOMAINS_SET" = true ] && [ "$WORDLIST_SET" = true ]; then
    # Fuzz subdomains
    mkdir -p fuzz_output
    while read subdomain; do
      FFUF_CMD="ffuf -w $WORDLIST -u http://${subdomain}/FUZZ"
      if [ "$RECURSIVE_FUZZ" = true ]; then
        FFUF_CMD="$FFUF_CMD -recursion -recursion-depth $FUZZ_DEPTH"
      fi
      $FFUF_CMD | tee "fuzz_output/directories_${subdomain}"
    done < "$SUBDOMAINS_FILE"
  else
    echo "Error: When -t is not specified, -s and -w are required for fuzzing."
    usage
  fi
fi

