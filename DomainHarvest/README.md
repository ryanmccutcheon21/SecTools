# DomainHarvest
DomainHarvest is a versatile domain enumeration tool designed to collect comprehensive information about a target domain from various sources.

## Features
    Fetches Wayback Machine URLs for the given domain.
    Retrieves common subdomains using crt.sh API.
    Options to customize output, such as showing only the domain without "https://," adding line breaks, or stripping paths.
    Extracts and sorts subdomains after running theHarvester.
    Merges all files created after running theHarvester.

## Download Instructions
To use DomainHarvest, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/ryanmccutcheon21/SecTools/DomainHarvest.git
2. **Navigate to the Tool Directory:**
```
cd DomainHarvest
```
3. **Install Dependencies:**
```
pip install -r requirements.txt
```
4. **Run the Tool:**
```
python3 DomainHarvest.py [options] target_domain
```
## For help and available options, use:
```
python3 DomainHarvest.py -i
```
## Options
    -d, --domain-only: Show only the domain without "https://."
    -l, --line-break: Add a line between each output.
    -s, --strip-path: Show only the domain without "https://" and without directories.
    -x, --extract: Extract and sort subdomains after running theHarvester.
    -m, --merge: Merge all files created after running theHarvester.
    -i, --help-info: Show help message.
    -o, --output-file: Specify an output file.
    -S, --sort: Sort the output alphabetically.
    -a, --all: Run -d, -s, -S, -x, and -m flags together.

## Examples
```
python3 DomainHarvest.py -s -S -o output.txt yahoo.com
```

## Requirements
    Python 3.x
    Requests library (install using pip install -r requirementst.txt)

## License
This project is licensed under the MIT License. Feel free to contribute or report issues!
