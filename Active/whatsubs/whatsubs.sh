#!/bin/bash

# Author: Ryan McCutcheon


# Check if the required tool 'whatweb' is installed
command -v whatweb >/dev/null 2>&1 || { echo >&2 "Please install 'whatweb'. Aborting."; exit 1; }

# Initialize variables with default values
output_dir="whatweb_output"
subdomains_file=""

# Function to show script usage
usage() {
    echo "Usage: $0 -s <subdomains_file> [-o <output_dir>]"
    echo "Options:"
    echo "  -s <subdomains_file>  Specify the path to the file containing subdomains."
    echo "  -o <output_dir>       Specify the output directory for result files. Default is 'whatweb_output'."
    exit 1
}

# Parse command line options
while getopts ":s:o:" opt; do
    case $opt in
        s)
            subdomains_file="$OPTARG"
            ;;
        o)
            output_dir="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# Check if the subdomains file is specified
if [ -z "$subdomains_file" ]; then
    echo "Error: Subdomains file not specified."
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate through each subdomain in the file
while IFS= read -r subdomain; do
    # Prepend 'https://' to the subdomain
    url="https://${subdomain}"

    # Define output file path
    output_file="${output_dir}/${subdomain}_whatweb.txt"

    echo "Scanning $url with whatweb..."

    # Use 'whatweb' to scan the subdomain and save the output to a file
    whatweb -a3 "$url" -v > "$output_file" 2>&1

    echo "Output saved to $output_file"

done < "$subdomains_file"

echo "Scan complete. Results saved in the '$output_dir' directory."

