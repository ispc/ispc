#!/usr/bin/env python3

# Copyright (c) 2025, Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import platform
import subprocess # nosec
import json
import shutil
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from multiprocessing import cpu_count


def get_llvm_asset(llvm_version, os_name, arch):
    """
    Find and retrieve LLVM release asset information based on the current system architecture
    and specified LLVM version from the ispc.dependencies GitHub repository.

    This function performs the following steps:
    1. Detects and normalizes the current OS and architecture to the form that
       is used in ispc.dependencies assets naming
    2. Queries GitHub API to find matching LLVM releases
    3. Locates the appropriate asset for the current platform
    4. Returns asset information for downloading

    Args:
        llvm_version (str): The desired LLVM version (e.g., "18", "17")
        os_name (str): The current operating system (e.g., "Linux", "Darwin", "Windows")
        arch (str): The current architecture (e.g., "x86_64", "aarch64")

    Returns:
        tuple: A tuple containing three elements:
            - asset_name (str): Name of the asset file
            - asset_url (str): Direct download URL for the asset
            - version (str): Extracted version number
            Or (None, None, None) if no matching asset is found or on API error

    Environment Requirements:
        - Supported operating systems: Linux (Ubuntu 22.04), macOS
        - Supported architectures: x86_64, aarch64 (Linux), arm64 (macOS)

    Example:
        >>> asset_name, asset_url, version = get_llvm_asset("18")
        Found release: llvm-18.0.0
        >>> print(asset_name)
        llvm-18.0.0-ubuntu22.04-Release-Assert-x86_64.tar.xz

    Notes:
        - The function uses GitHub's public API and may be subject to rate limiting
        - For Linux arm64, the asset name includes 'aarch64'
        - For macOS arm64, the asset name does not include architecture
        - The function excludes LTO (Link Time Optimization) variants of assets
    """
    # Normalize OS names
    os_map = {
        'Linux': 'ubuntu22.04',
        'Darwin': 'macos',
        'Windows': 'win'
    }

    if os_name not in os_map:
        raise RuntimeError(f"Unsupported OS: {os_name}")

    os_name = os_map[os_name]

    # Normalize architecture names
    if arch == 'x86_64' or arch == 'AMD64':
        arch = ''
    elif arch == 'aarch64':
        arch = 'aarch64'  # linux tarball contains aarch64 after OS
    elif arch == 'arm64':
        arch = ''  # macOS specific, tarballs don't contain arm in their names
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    # Fetch GitHub releases
    headers = {'User-Agent': 'Python'}
    try:
        req = Request("https://api.github.com/repos/ispc/ispc.dependencies/releases", headers=headers)
        with urlopen(req) as response: # nosec
            releases_json = json.loads(response.read())
    except HTTPError as e:
        if "rate limit exceeded" in str(e):
            print("GitHub API rate limit exceeded.")
            return None, None, None
        raise

    # Find matching release
    matching_release = None
    for release in releases_json:
        if release['tag_name'].startswith(f"llvm-{llvm_version}."):
            matching_release = release['tag_name']
            break

    if not matching_release:
        print(f"No matching release found for llvm-{llvm_version}.*")
        return None, None, None

    print(f"Found release: {matching_release}")
    version = matching_release[5:].split('-')[0]  # Remove 'llvm-' prefix and everything after first '-'

    # Fetch assets for the matching release
    try:
        req = Request(f"https://api.github.com/repos/ispc/ispc.dependencies/releases/tags/{matching_release}", headers=headers)
        with urlopen(req) as response: # nosec
            assets_json = json.loads(response.read())
    except HTTPError as e:
        if "rate limit exceeded" in str(e):
            print("GitHub API rate limit exceeded.")
            return None, None, None
        raise

    # Find matching asset
    asset_pattern = f"llvm-{llvm_version}.*-{os_name}{arch}-Release.*Asserts-.*\\.tar\\.xz"
    if os_name == "win":
        asset_pattern = f"llvm-{llvm_version}.*-{os_name}.*-Release.*Asserts-.*\\.tar\\.7z"
    for asset in assets_json['assets']:
        if re.match(asset_pattern, asset['name']) and 'lto' not in asset['name']:
            return asset['name'], asset['browser_download_url'], version

    print(f"No matching assets found for release {matching_release} and pattern {asset_pattern}")
    return None, None, None


def download_file(url, filename):
    """Download a file from a URL to a local destination with progress indication.

    This function downloads a file from the specified URL and saves it locally,
    displaying a progress bar when the content length is available. It handles
    both cases where content length is known and unknown.

    Args:
        url (str): The URL of the file to download. Must be a valid URL that points
            to the target file. Supports HTTP and HTTPS protocols.
        filename (str): The local path where the downloaded file will be saved.
            If the path doesn't exist, intermediary directories will not be created.

    Returns:
        None

    Examples:
        >>> # Download a file with known content length
        >>> download_file('https://example.com/file.zip', 'local_file.zip')
        Downloading: 23%
        Downloading: 47%
        Downloading: 98%
        Downloading: 100%

        >>> # Download a file with unknown content length
        >>> download_file('https://example.com/stream', 'local_stream.dat')

    Notes:
        - Uses a custom User-Agent header to identify the client as Python
        - For files with known size, downloads in chunks of 8192 bytes
        - Progress indication is only shown when Content-Length header is present
        - Progress updates are written to stdout with carriage return for in-place updates
        - Falls back to shutil.copyfileobj() for streams with unknown length

    Warning:
        Ensure you have write permissions in the target directory and sufficient
        disk space before downloading large files.
    """
    headers = {'User-Agent': 'Python'}
    req = Request(url, headers=headers)
    with urlopen(req) as response, open(filename, 'wb') as out_file: # nosec
        content_length = response.headers.get('Content-Length')
        if content_length:
            total_size = int(content_length)
            downloaded = 0
            chunk_size = 64 * 1024
            last_progress = -1  # Track the last printed progress

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                downloaded += len(chunk)
                out_file.write(chunk)

                # Calculate progress as an integer percentage
                progress = int((downloaded / total_size) * 100)
                if progress != last_progress:
                    print(f"\rDownloading: {progress}%", end='', flush=True)
                    last_progress = progress
            print()  # New line after finishing
        else:
            # Fallback for streams with unknown content length
            shutil.copyfileobj(response, out_file)


def extract_archive(archive_path, is_windows):
    """Extract an archive file with special handling for Windows .tar.7z files.

    This function handles archive extraction with different behavior based on the
    operating system. For Windows, it specifically handles .tar.7z files using
    py7zr, performing a two-step extraction process. For other platforms, it
    directly extracts tar archives.

    Args:
        archive_path (str or Path): Path to the archive file to extract.
            For Windows: Expected to be a .tar.7z file
            For other OS: Expected to be a .tar file
        is_windows (bool): Flag indicating if the current OS is Windows.
            True for Windows systems, False for other operating systems.

    Returns:
        None

    Examples:
        >>> # On Windows with a .tar.7z file
        >>> extract_archive('llvm-13.0.0.tar.7z', True)
        Extracting llvm-13.0.0.tar.7z
        Extracting llvm-13.0.0.tar

        >>> # On Linux/Mac with a .tar file
        >>> extract_archive('llvm-13.0.0.tar', False)
        Extracting llvm-13.0.0.tar

    Notes:
        - On Windows:
          1. First extracts the outer .7z container
          2. Then extracts the inner .tar file
          3. Requires py7zr package to be installed
        - On other platforms:
          1. Directly extracts the .tar file
        - Extracts all contents to the current working directory
        - Does not preserve original archive file

    Warning:
        - Ensure sufficient disk space for extraction
        - Extraction overwrites existing files without confirmation
        - On Windows, temporary .tar file is created during extraction
        - Current directory must be writable

    Dependencies:
        - py7zr (Windows only): Required for .7z extraction
        - tarfile: Built-in Python module for tar extraction
        - pathlib: For Path operations
    """
    if is_windows:
        try:
            import py7zr
        except ImportError:
            print("Error: py7zr is required for extracting .tar.7z files on Windows.")
            print("Please install it with 'pip install py7zr'.")
            sys.exit(1)

        # For Windows .tar.7z files
        print(f"Extracting {archive_path}")
        # First extract the .7z
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall()
        # Then extract the resulting .tar
        archive_path = next(Path('.').glob('llvm*.tar'))

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path) as tar:
        tar.extractall()


def run_command(cmd, on_error=None, env=None):
    """Execute a shell command and handle any failures.

    Runs a subprocess with the given command and checks its return code.
    If the command fails, executes an optional error callback and exits with
    the same return code.

    Args:
        cmd (list): Command to execute as a list of strings or Path objects.
            Example: ['cmake', '--build', 'path/to/build', '--target', 'all']
        on_error (callable, optional): Function to call if command fails.
            Will be called with no arguments before program exit.
            Default: None
        env (dict, optional): Environment variables for the subprocess.
            Passed directly to subprocess.run().
            If None, uses current environment.
            Default: None

    Returns:
        subprocess.CompletedProcess: If command executes successfully

    Examples:
        >>> # Basic usage
        >>> run_command(['echo', 'test'])
        test
        <CompletedProcess(args=['echo', 'test'], returncode=0)>

        >>> # With error callback
        >>> def cleanup(): print("Cleaning up...")
        >>> run_command(['false'], on_error=cleanup)
        Command failed with exit code 1: false
        Cleaning up...
        # Exits program with code 1

        >>> # With custom environment
        >>> run_command(['printenv', 'CUSTOM'], env={'CUSTOM': 'value'})
        value

    Notes:
        - Uses subprocess.run with default settings (no shell, no output capture)
        - Command output goes directly to stdout/stderr
        - All arguments are converted to strings using str()
        - Error callback runs before program termination

    Warning:
        This function will terminate the entire program if the command fails.
        Use subprocess.run directly if you need to handle command failures differently.
    """
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env) if env else subprocess.run(cmd)
    if result.returncode:
        print(f"Command failed with exit code {result.returncode}: {' '.join(map(str, cmd))}")
        if on_error:
            on_error()
        sys.exit(result.returncode)


def main():
    """Set up and build ISPC with specified LLVM version.

    This is the main entry point for the ISPC build script. It handles downloading
    and setting up LLVM dependencies, configuring the build environment, and
    running the ISPC build and test suite.

    Environment Variables:
        LLVM_HOME (str, optional): Directory for LLVM installation.
            Defaults to current working directory.
        ISPC_HOME (str, optional): Root directory of ISPC source.
            Defaults to parent directory of this script.
        ARCHIVE_URL (str, optional): Direct download URL for LLVM package.
            Required only if automatic asset detection fails.

    Command Line Args:
        llvm_version (str, optional): LLVM version to use.
            Default: "18"

    Directory Structure Created/Used:
        Working Directory (LLVM_HOME)/
            - llvm-{version}/      # LLVM installation (downloaded and extracted)
        ISPC Build Directory (ISPC_HOME)/
            - build-{version}/     # ISPC build directory
        ISPC Root Directory is determined by the script location

    Build Process:
        1. Determines system configuration (OS, architecture, CPU count)
        2. Downloads and extracts LLVM if not present
        3. Configures ISPC build with CMake
        4. Builds ISPC with parallel compilation
        5. Runs ISPC support matrix check
        6. Executes test suite

    Returns:
        None

    Examples:
        # Build with default LLVM 18
        $ python quick-start-build.py

        # Build with specific LLVM version
        $ python quick-start-build.py 17

        # Build with custom LLVM location
        $ LLVM_HOME=/path/to/llvm python quick-start-build.py

        # Build with custom LLVM location Windows cmd
        > set LLVM_HOME=C:\path\to\llvm && python quick-start-build.py

        # Build with custom LLVM location PowerShell
        > $env:LLVM_HOME = "C:\path\to\llvm"; python quick-start-build.py


    Build Configurations:
        Windows:
            - Build Type: RelWithDebInfo
            - Binary Location: build-{version}/bin/RelWithDebInfo/ispc

        Other Platforms:
            - Build Type: Debug
            - Binary Location: build-{version}/bin/ispc

    Notes:
        - Automatically determines optimal parallel build count
        - Cleans up failed CMake configurations
        - Preserves existing LLVM and build directories if present
        - Uses GitHub API to fetch appropriate LLVM binaries
        - Supports custom LLVM archive URLs via environment variable

    Dependencies:
        - CMake: For building ISPC
        - Python 3.6+: For script execution
        - C++ Compiler: Compatible with chosen LLVM version
        - m4: Required for ISPC build
        - flex, bison: Required for ISPC build
        - onetbb: Required for ISPC build
    """
    # Set up default values and paths
    llvm_version = sys.argv[1] if len(sys.argv) > 1 else "18"
    llvm_home = os.getenv("LLVM_HOME", os.getcwd())
    # Determine number of processors (defaulting to 8 if unknown)
    try:
        nproc = cpu_count() or 8
    except Exception:
        nproc = 8

    scripts_dir = Path(__file__).parent.absolute()
    ispc_root = scripts_dir.parent.absolute()
    ispc_home = Path(os.getenv("ISPC_HOME", str(ispc_root)))
    build_dir = ispc_home / f"build-{llvm_version}"
    llvm_dir = Path(llvm_home) / f"llvm-{llvm_version}"

    arch = platform.machine()
    os_name = platform.system()
    is_windows = os_name == 'Windows'

    print(f"LLVM_HOME: {llvm_home}")
    print(f"ISPC_HOME: {ispc_home}")

    os.chdir(llvm_home)

    if not llvm_dir.exists():
        asset_name, asset_url, version = get_llvm_asset(llvm_version, os_name, arch)

        if not asset_name:
            archive_url = os.getenv("ARCHIVE_URL")
            if archive_url:
                asset_name = os.path.basename(archive_url)
                asset_url = archive_url
                version = re.search(f"{llvm_version}\\.[0-9]*", asset_name).group(0)
            else:
                print("Error: Failed to deduct and fetch LLVM archives from Github API.")
                print("Please set ARCHIVE_URL environment variable to the direct download URL of the LLVM package.")
                print("Example: export ARCHIVE_URL='https://github.com/ispc/ispc.dependencies/releases/download/...'")
                sys.exit(1)

        print(f"Asset Name: {asset_name}")
        print(f"Download URL: {asset_url}")

        if Path(asset_name).exists():
            Path(asset_name).unlink()

        download_file(asset_url, asset_name)
        extract_archive(asset_name, is_windows)

        Path(f"bin-{version}").rename(llvm_dir)
    else:
        print(f"{llvm_dir} already exists")

    if not build_dir.exists():
        env = os.environ.copy()
        env["PATH"] = f"{llvm_dir / 'bin'}{os.pathsep}{env['PATH']}"

        build_type = "RelWithDebInfo" if is_windows else "Debug"
        configure_cmd = [
            "cmake",
            "-B", str(build_dir),
            str(ispc_root),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DISPC_SLIM_BINARY=ON"
        ]
        print("Configure build of ISPC")
        run_command(configure_cmd,
                    lambda: (
                        print(f"CMake failed, cleaning up build directory {build_dir}"),
                        shutil.rmtree(build_dir)
                        ),
                    env=env)
    else:
        print(f"{build_dir} already exists")

    print("Build ISPC")
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel", str(nproc)]
    if is_windows:
        build_cmd.extend(["--config", build_type])
    run_command(build_cmd)

    print("Run ispc --support-matrix")
    ispc_bin = build_dir / "bin"
    ispc_exe = ispc_bin / build_type / "ispc" if is_windows else ispc_bin / "ispc"
    run_command([str(ispc_exe), "--support-matrix"])

    print("Run check-all")
    check_all_cmd = ["cmake", "--build", str(build_dir), "--target", "check-all"]
    if is_windows:
        check_all_cmd.extend(["--config", build_type])
    run_command(check_all_cmd)

if __name__ == "__main__":
    main()
