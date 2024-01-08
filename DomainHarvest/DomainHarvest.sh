#!/bin/bash

usage() {
  echo "Usage: $0 -t <target> [-l] [-c] [-h]"
  echo "Options:"
  echo "  -t <target>        Set the target domain."
  echo "  -l                Show all subdomains."
  echo "  -c                Show the count of domains found."
  echo "  -h                Show this help message with tool introduction."
  exit 1
}

tool_intro() {
  echo "Custom Domain Enumeration Tool"
  echo "This tool automates the process of collecting subdomains using various sources."
  echo "It supports multiple functionalities such as running theHarvester, extracting, and sorting subdomains."
  echo "Options:"
  echo "  -t <target>        Set the target domain."
  echo "  -l                Show all subdomains."
  echo "  -c                Show the count of domains found."
  echo "  -h                Show this help message with tool introduction."
  exit 0
}

# Step 1: Create sources.txt
touch sources.txt

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

# Parse command line options
while getopts ":t:lc:h" opt; do
  case $opt in
    t)
      TARGET="$OPTARG"
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
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done

if [ -z "$TARGET" ]; then
  echo "Error: Please provide a target domain."
  usage
fi

# Step 2: Make files directory
mkdir files

# Step 3: Run theHarvester
cat sources.txt | while read source; do
  theHarvester -d "${TARGET}" -b $source -f "./files/${source}_${TARGET}"
done

# Step 4: Extract and sort subdomains
cat ./files/*.json | jq -r '.hosts[]' 2>/dev/null | cut -d':' -f 1 | sort -u > "./files/${TARGET}_theHarvester.txt"

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

