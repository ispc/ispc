# Copyright 2024, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# This script performs DLL injection checks by creating an empty DLL,
# monitoring process activity using Procmon, and filtering events related
# to the specified DLL. It ensures that the DLL is not loaded from the
# current directory and cleans up temporary files after execution.

import os
import time
# Subprocess is used with default shell which is False, it's safe and doesn't allow shell injection
# so we can ignore the Bandit warning
import subprocess #nosec
# defusedxml.defuse_stdlib() is called to defuse the standard library
import xml.etree.ElementTree as ET #nosec
import argparse
import sys
import winreg
from defusedxml import defuse_stdlib

defuse_stdlib()  # Call to defuse the standard library

def create_empty_dll(dll_name):
    print(f"Creating empty DLL: {dll_name}")
    dll_path = os.path.join(os.getcwd(), dll_name)
    if not os.path.exists(dll_path):
        with open(dll_path, 'w') as f:
            pass
        print(f"Empty {dll_name} created in the current directory.")
    return dll_path.lower()

def check_path(command_name):
    print(f"Checking path for: {command_name}")
    if subprocess.call(f"where {command_name}", stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
        print(f"Error: {command_name} not found in the system PATH.")
        exit(1)

def start_procmon(logfile):
    print(f"Starting Procmon with logfile: {logfile}")
    subprocess.Popen(["Procmon.exe", "/BackingFile", logfile, "/AcceptEula", "/Quiet"])
    time.sleep(10)  # Initial wait

def run_ispc(times=3):
    for i in range(times):
        print(f"Running ispc.exe... (Attempt {i + 1})")
        process = subprocess.Popen(["ispc.exe"])
        time.sleep(2)
        print("Terminating ispc.exe...")
        process.terminate()  # Terminate the process if it's still alive
        process.wait()  # Wait for the process to terminate

def export_log_to_xml(logfile, xml_log):
    # Export the log from Procmon to an XML file for further analysis.
    subprocess.run(["Procmon.exe", "/OpenLog", logfile, "/SaveAs", xml_log], check=True)
    time.sleep(30)

def filter_events(xml_log, dll_name):
    # Filter the events in the XML log to find those related to the specified DLL.
    filtered_events = []
    # defusedxml.defuse_stdlib() is called to defuse the standard library
    tree = ET.parse(xml_log) #nosec
    root = tree.getroot()

    for event in root.iter('event'):
        process_name = event.find('Process_Name').text if event.find('Process_Name') is not None else None
        path = event.find('Path').text if event.find('Path') is not None else None

        if (process_name == "ispc.exe" and path and
            (path.lower().endswith(dll_name) or dll_name in path.lower())):
            filtered_events.append(event)

    return filtered_events

def save_filtered_events(filtered_events, dll_load_filtered):
    # Save the filtered events to a new XML file for reporting purposes.
    filtered_tree = ET.Element("ProcessMonitor")

    for event in filtered_events:
        event_node = ET.SubElement(filtered_tree, "event")
        for child in event:
            child_node = ET.SubElement(event_node, child.tag)
            child_node.text = child.text

    filtered_tree = ET.ElementTree(filtered_tree)
    filtered_tree.write(dll_load_filtered)
    print(f"Filtered log file created at: {dll_load_filtered}")

def cleanup_temp_files(dll_path, logfile, xml_log):
    print("Cleaning up temporary files...")
    if os.path.exists(dll_path):
        os.remove(dll_path)
    if os.path.exists(logfile):
        os.remove(logfile)
    if os.path.exists(xml_log):
        os.remove(xml_log)
    print("Temporary files cleaned up.")

def set_safe_dll_search_mode_off():
    # Safe DLL search mode (which is enabled by default) moves the user's current folder later in the search order.
    # Let's turn it off.
    print("Setting SafeDllSearchMode registry value to 0...")
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager", 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "SafeDllSearchMode", 0, winreg.REG_DWORD, 0)
        print("SafeDllSearchMode set to 0 successfully.")
    except Exception as e:
        print(f"Failed to set SafeDllSearchMode: {e}")

def main():
    if sys.platform != "win32":
        print("Error: This script can only be executed on Windows.")
        exit(1)
    parser = argparse.ArgumentParser(description='DLL Injection Checker')
    parser.add_argument('dll_name', type=str, help='Name of the DLL to be injected')
    args = parser.parse_args()

    print(f"Arguments received: {args.dll_name}")

    dll_path = create_empty_dll(args.dll_name)

    check_path("ispc.exe")
    check_path("Procmon.exe")

    set_safe_dll_search_mode_off()

    logfile = os.path.join(os.getcwd(), "procmon_log.pml")
    xml_log = os.path.join(os.getcwd(), "procmon_log.xml")
    dll_load_filtered = os.path.join(os.getcwd(), "dll_load_filtered.xml")

    start_procmon(logfile)
    run_ispc()
    subprocess.run(["Procmon.exe", "/Terminate"], check=True)
    time.sleep(30)

    if not os.path.exists(logfile):
        print("Error: Log file was not created.")
        exit(1)

    export_log_to_xml(logfile, xml_log)

    if not os.path.exists(xml_log):
        print("Error: XML log file was not created.")
        exit(1)

    print("Filtering the results...")
    filtered_events = filter_events(xml_log, args.dll_name)

    dll_loaded_from_current_dir = False
    # Check if any relevant events were found for the specified DLL
    if not filtered_events:
        print("No relevant events found for ispc.exe loading ", args.dll_name)
    else:
        # Determine if the DLL was loaded from the current directory
        dll_loaded_from_current_dir = any(event.find('Path').text.lower() == dll_path for event in filtered_events)

        if dll_loaded_from_current_dir:
            print("Error: " + args.dll_name + " was loaded from the current directory.")
        else:
            print(args.dll_name + " loaded from system directory or other location. Continuing...")

        # Save the filtered events to an XML file for reporting
        save_filtered_events(filtered_events, dll_load_filtered)

    # Clean up temporary files created during the process
    cleanup_temp_files(dll_path, logfile, xml_log)

    # Return 1 if the DLL was loaded from the current directory, else return 0
    return 1 if dll_loaded_from_current_dir else 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)