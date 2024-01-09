#!/bin/bash

# Subheaders - Enumerate through a list of subdomains and fetch HTTP headers using curl.
# Author: Ryan McCutcheon
#GitHub: ryanmccutcheon21

# Disclaimer:
# This tool is provided for educational and informational purposes only. Use it responsibly and ensure that you have proper authorization before testing it on any network or system

# Initialize variables with default values
output_dir="headers"
user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
follow_redirects=false
timeout=5  # Default timeout value in seconds

# Function to show script usage
usage() {
    echo "Usage: $0 -sub <subdomains_file> [-o <output_dir>] [-a <user_agent>] [-r] [-t <timeout>]"
    echo "Options:"
    echo "  -sub <subdomains_file>  Specify the subdomains file to enumerate through."
    echo "  -o <output_dir>         Specify the output directory for result files. Default is 'headers'."
    echo "  -a <user_agent>         Specify the User-Agent string. Default is a common browser user agent."
    echo "  -r                      Follow redirects when making requests (curl -L)."
    echo "  -t <timeout>            Specify curl timeout in seconds. Default is $timeout seconds."
    exit 1
}

# Parse command line options
while [ "$#" -gt 0 ]; do
    case "$1" in
        -sub)
            subdomains_file="$2"
            shift 2
            ;;
        -o)
            output_dir="$2"
            shift 2
            ;;
        -a)
            user_agent="$2"
            shift 2
            ;;
        -r)
            follow_redirects=true
            shift
            ;;
        -t)
            timeout="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage
            ;;
    esac
done

# Check if the subdomains file is specified
if [ -z "$subdomains_file" ]; then
    echo "Error: Subdomains file not specified."
    usage
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Read each line from the subdomains file and call curl
while IFS= read -r subdomain; do
    output_file="${output_dir}/${subdomain}_headers.txt"
    url="https://${subdomain}"

    echo "Checking $url..."
    
    # Use curl to fetch headers and write them to a file, with a timeout
    curl_cmd="curl -s -I -A '$user_agent' --max-time $timeout"
    [ "$follow_redirects" = true ] && curl_cmd="$curl_cmd -L"
    curl_output="$($curl_cmd $url)"

    # Check if there is output from the curl command
    if [ -n "$curl_output" ]; then
        echo "$curl_output" > "$output_file"
        echo "Headers saved for $subdomain"
    else
        echo "No response from $url. Skipping..."
    fi

    # You can add more processing or output as needed
done < "$subdomains_file"

echo "Done. Results saved in the '$output_dir' directory."

