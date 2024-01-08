import requests
import argparse
from urllib.parse import urlparse

def fetch_wayback_urls(domain):
    wayback_api_url = f"https://web.archive.org/cdx/search/cdx?url=*.{domain}/*&output=json&fl=original&collapse=urlkey"
    response = requests.get(wayback_api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return []

def fetch_common_subdomains(domain):
    crtsh_api_url = f"https://crt.sh/?q=%.{domain}&output=json"
    response = requests.get(crtsh_api_url)

    if response.status_code == 200:
        try:
            subdomains = [entry['name_value'] for entry in response.json()]
            return subdomains
        except requests.exceptions.JSONDecodeError:
            return []
    else:
        return []

def main():
    parser = argparse.ArgumentParser(description="Custom domain enumeration tool")
    parser.add_argument("domain", nargs="?", help="Target domain to enumerate")
    parser.add_argument("-d", "--domain-only", action="store_true", help="Show only the domain without https://")
    parser.add_argument("-l", "--line-break", action="store_true", help="Add a line between each output")
    parser.add_argument("-s", "--strip-path", action="store_true", help="Show only the domain without https:// and without directories and subdirectories")
    parser.add_argument("-i", "--help-info", action="store_true", help="Show information about how to use the tool")
    parser.add_argument("-o", "--output-file", help="Specify an output file")
    parser.add_argument("-S", "--sort", action="store_true", help="Sort the output alphabetically")

    args = parser.parse_args()

    if args.help_info:
        parser.print_help()
        print("\nExplanation of Flags:")
        print("-d, --domain-only: Show only the domain without https://")
        print("-l, --line-break: Add a line between each output")
        print("-s, --strip-path: Show only the domain without https:// and without directories and subdirectories")
        print("-i, --help-info: Show this help message")
        print("-o, --output-file: Specify an output file")
        print("-S, --sort: Sort the output alphabetically")
        exit()

    if not args.domain and not args.help_info:
        print("Error: Please provide a domain.")
        parser.print_help()
        exit(1)

    domain = args.domain
    show_domain_only = args.domain_only
    add_line_break = args.line_break
    strip_path = args.strip_path

    wayback_urls = fetch_wayback_urls(domain)
    subdomains = fetch_common_subdomains(domain)

    all_urls_set = set()

    for url in wayback_urls:
        all_urls_set.add(url[0])

    for subdomain in subdomains:
        all_urls_set.add(subdomain)

    output_list = list(all_urls_set)

    if args.sort:
        output_list.sort()

    if args.output_file:
        with open(args.output_file, 'w') as output_file:
            for item in output_list:
                if strip_path:
                    parsed_url = urlparse(item)
                    output_file.write(f"{parsed_url.netloc}\n")
                elif show_domain_only:
                    output_file.write(f"{item.split('://')[1] if '://' in item else item}\n")
                else:
                    output_file.write(f"{item}\n")
    else:
        for item in output_list:
            if strip_path:
                parsed_url = urlparse(item)
                print(parsed_url.netloc)
            elif show_domain_only:
                print(item.split("://")[1] if "://" in item else item)
            else:
                print(item)

            if add_line_break:
                print()

if __name__ == "__main__":
    main()

