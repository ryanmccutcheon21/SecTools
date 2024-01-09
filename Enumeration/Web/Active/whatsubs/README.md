# WhatSubs - Subdomain Scanner with WhatWeb

## Overview
WhatSubs is a bash script that utilizes the 'whatweb' tool to scan subdomains from a list and save the results for each subdomain.

## Prerequisites
- [WhatWeb](https://github.com/urbanadventurer/WhatWeb): Ensure 'whatweb' is installed on your system.

## Usage
```
whatsubs -s <subdomains_file> [-o <output_dir>]
```

## Options:
```
    -s <subdomains_file>: Specify the path to the file containing subdomains.
    -o <output_dir>: Specify the output directory for result files. Default is 'whatsubs_output'.
```

## Download and Install
### Clone the repository:
```
git clone https://github.com/ryanmccutcheon21/SecTools/Active/whatsubs.git
```

### Navigate to the directory:
```
cd whatsubs
```
### Make the script executable:
```
chmod +x whatsubs.sh
```

### Create a Symbolic Link:
```
sudo ln -s /$(pwd)/whatsubs.sh /usr/local/bin/whatsubs
```

### Run the script:
```
    whatsubs -h
```

## Example
```
whatsubs -s subdomains.txt -o whatsubs_output
```
Results will be saved in the 'whatsubs_output' directory.


## Disclaimer
Use responsibly and ensure compliance with laws and regulations. The tool is provided as-is, without any warranties or guarantees.

Author
Ryan McCutcheon
