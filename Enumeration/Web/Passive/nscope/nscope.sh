#!/bin/bash

# Function to filter out-of-scope URLs
filter_urls() {
    local out_of_scope_file="$1"
    local filtered_urls_file="$2"
    local blacklisted_urls_file="$3"

    # Extract anything that resembles a URL from out_of_scope_file
    grep -Eo '(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})? ' "$out_of_scope_file" > /tmp/temp_filtered_urls

    # Filter out blacklisted URLs
    grep -vFf "$blacklisted_urls_file" /tmp/temp_filtered_urls > out_of_scope_domains.txt
}

# Function to filter subdomains
filter_subdomains() {
    local subdomains_file="$1"
    local filtered_urls_file="$2"
    local blacklisted_urls_file="$3"

    # Filter subdomains based on the blacklisted URLs
    grep -Ff "out_of_scope_domains.txt" "$subdomains_file" > removed.txt

    # Filter subdomains based on the out_of_scope_domains.txt
    grep -vFf "out_of_scope_domains.txt" "$subdomains_file" > nscope.txt

    echo "Removed subdomains saved to: removed.txt"
    echo "In-scope subdomains saved to: nscope.txt"
}

# Function to display help information
show_help() {
    echo "==================================================================================================================="
    echo "Tool that takes a text file that you can copy and paste from your desired target's bug bounty out-of-scope page, and filters the text to gather the domains that are out-of-scope. It then compares these domains to your file containing domains found during recon and filters the out-of-scope domains. The removed domains will be output in the file specified with the -r flag, and the in-scope domains will be output to the file specified with the -f flag."
    echo "===================================================================================================================="
    echo "Usage: nscope  -o <out_of_scope_file> -s <subdomains_file> -b <blacklisted_urls_file> -h"
    echo "===================================================================================================================="
    echo "Options:"
    echo "  -o <out_of_scope_file>        Specify the file containing out-of-scope information."
    echo "  -s <subdomains_file>          Specify the file containing subdomains."
    echo "  -b <blacklisted_domains_file> Specify the file containing domains you specifically do not want removed from the subdomains file specified with the -s flag."
    echo "  -h                            Show this help message."
    echo "===================================================================================================================="
}

# Parse command line options
while getopts ":o:s:b:h" opt; do
    case $opt in
        o)
            out_of_scope_file="$OPTARG"
            ;;
        s)
            subdomains_file="$OPTARG"
            ;;
        b)
            blacklisted_urls_file="$OPTARG"
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

# Check if mandatory options are provided
if [ -z "$out_of_scope_file" ] || [ -z "$blacklisted_urls_file" ]; then
    echo "Error: Missing required options. See help (-h) for usage."
    exit 1
fi

# Call the filter_urls function
filter_urls "$out_of_scope_file" "filtered_domains.txt" "$blacklisted_urls_file"

# Check if subdomains_file is provided
if [ -n "$subdomains_file" ]; then
    # Call the filter_subdomains function
    filter_subdomains "$subdomains_file" "filtered_domains.txt" "$blacklisted_urls_file"
fi

rm /tmp/temp_filtered_urls

