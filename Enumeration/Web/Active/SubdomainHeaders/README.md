# Subheaders
The subheaders tool is a bash script designed for efficiently enumerating through a list of subdomains and fetching HTTP headers using curl. It provides flexibility in customizing the user agent, following redirects, and specifying a timeout for each request.

## Features:
    Enumerate through a list of subdomains.
    Fetch HTTP headers for each subdomain using curl.
    Customize user agent, output directory, and follow redirects.
    Specify a timeout for each request.

## Usage:
```
subheaders -sub <subdomains_file> [-o <output_dir>] [-a <user_agent>] [-r] [-t <timeout>]
```

## Download and Install Instructions:
### Download the Script:
Clone the GitHub repository or download the subheaders.sh file.
```
git clone https://github.com/ryanmccutcheon21/SecTools/Active/SubdomainHeaders.git
```

### Make the Script Executable:
Ensure that the script has execution permissions.
```
chmod +x subheaders.sh
```

### Create a Symbolic Link:
Create a symbolic link to the script in a directory that is included in your PATH. For example, /usr/local/bin/ is a common directory.
```
    sudo ln -s /path/to/subheaders.sh /usr/local/bin/subheaders
```
Replace /path/to/subheaders.sh with the actual path to your subheaders.sh file.

### Verify Installation:
Open a new terminal window and run the following command to verify the installation.
```
subheaders -h
```
This should display the help message for the subheaders tool.

