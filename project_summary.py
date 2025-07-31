
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: project_summary.py
# execution: true
import os
import json

def check_file_exists(file_path):
    """Check if a file exists and return its size."""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        return True, size
    return False, 0

def check_directory_exists(dir_path):
    """Check if a directory exists and count its files."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        return True, file_count
    return False, 0

def get_project_status():
    """Get the status of the project implementation."""
    # Core components
    core_files = [
        "promptshield/__init__.py",
        "promptshield/main.py",
        "promptshield/router.py",
        "promptshield/classifier.py",
        "promptshield/cache.py",
        "promptshield/sdk.py",
        "promptshield/cli.py",
    ]
    
    # Client adapters
    client_files = [
        "promptshield/clients/__init__.py",
        "promptshield/clients/openai.py",
        "promptshield/clients/anthropic.py",
        "promptshield/clients/ollama.py",
        "promptshield/clients/vllm.py",
    ]
    
    # Configuration and setup
    config_files = [
        "config.yaml",
        "setup.py",
        "README.md",
        "requirements.txt",
    ]
    
    # Test files
    test_files = [
        "tests/__init__.py",
        "tests/test_classifier.py",
        "tests/test_router.py",
        "tests/test_cache.py",
    ]
    
    # Check core components
    core_status = {}
    for file in core_files:
        exists, size = check_file_exists(file)
        core_status[file] = {"exists": exists, "size": size}
    
    # Check client adapters
    client_status = {}
    for file in client_files:
        exists, size = check_file_exists(file)
        client_status[file] = {"exists": exists, "size": size}
    
    # Check configuration and setup
    config_status = {}
    for file in config_files:
        exists, size = check_file_exists(file)
        config_status[file] = {"exists": exists, "size": size}
    
    # Check test files
    test_status = {}
    for file in test_files:
        exists, size = check_file_exists(file)
        test_status[file] = {"exists": exists, "size": size}
    
    # Check directories
    directory_status = {}
    directories = ["promptshield", "promptshield/clients", "promptshield/logs", "promptshield/dashboard", "tests"]
    for directory in directories:
        exists, file_count = check_directory_exists(directory)
        directory_status[directory] = {"exists": exists, "file_count": file_count}
    
    # Calculate completion percentages
    core_completion = sum(1 for file in core_status.values() if file["exists"]) / len(core_status) * 100
    client_completion = sum(1 for file in client_status.values() if file["exists"]) / len(client_status) * 100
    config_completion = sum(1 for file in config_status.values() if file["exists"]) / len(config_status) * 100
    test_completion = sum(1 for file in test_status.values() if file["exists"]) / len(test_status) * 100
    directory_completion = sum(1 for dir in directory_status.values() if dir["exists"]) / len(directory_status) * 100
    
    overall_completion = (core_completion + client_completion + config_completion + test_completion + directory_completion) / 5
    
    return {
        "core_components": {
            "status": core_status,
            "completion": core_completion
        },
        "client_adapters": {
            "status": client_status,
            "completion": client_completion
        },
        "configuration": {
            "status": config_status,
            "completion": config_completion
        },
        "tests": {
            "status": test_status,
            "completion": test_completion
        },
        "directories": {
            "status": directory_status,
            "completion": directory_completion
        },
        "overall_completion": overall_completion
    }

def print_project_summary():
    """Print a summary of the project status."""
    status = get_project_status()
    
    print("PromptShield Project Status Summary")
    print("===================================")
    print(f"Overall Completion: {status['overall_completion']:.1f}%")
    print()
    
    print("Core Components:")
    for file, info in status["core_components"]["status"].items():
        status_str = "✅" if info["exists"] else "❌"
        size_str = f"{info['size']} bytes" if info["exists"] else "N/A"
        print(f"  {status_str} {file} ({size_str})")
    print(f"  Completion: {status['core_components']['completion']:.1f}%")
    print()
    
    print("Client Adapters:")
    for file, info in status["client_adapters"]["status"].items():
        status_str = "✅" if info["exists"] else "❌"
        size_str = f"{info['size']} bytes" if info["exists"] else "N/A"
        print(f"  {status_str} {file} ({size_str})")
    print(f"  Completion: {status['client_adapters']['completion']:.1f}%")
    print()
    
    print("Configuration and Setup:")
    for file, info in status["configuration"]["status"].items():
        status_str = "✅" if info["exists"] else "❌"
        size_str = f"{info['size']} bytes" if info["exists"] else "N/A"
        print(f"  {status_str} {file} ({size_str})")
    print(f"  Completion: {status['configuration']['completion']:.1f}%")
    print()
    
    print("Tests:")
    for file, info in status["tests"]["status"].items():
        status_str = "✅" if info["exists"] else "❌"
        size_str = f"{info['size']} bytes" if info["exists"] else "N/A"
        print(f"  {status_str} {file} ({size_str})")
    print(f"  Completion: {status['tests']['completion']:.1f}%")
    print()
    
    print("Directories:")
    for directory, info in status["directories"]["status"].items():
        status_str = "✅" if info["exists"] else "❌"
        file_count_str = f"{info['file_count']} files" if info["exists"] else "N/A"
        print(f"  {status_str} {directory} ({file_count_str})")
    print(f"  Completion: {status['directories']['completion']:.1f}%")
    print()
    
    print("Next Steps:")
    if status["overall_completion"] < 100:
        missing_components = []
        
        for file, info in status["core_components"]["status"].items():
            if not info["exists"]:
                missing_components.append(file)
        
        for file, info in status["client_adapters"]["status"].items():
            if not info["exists"]:
                missing_components.append(file)
        
        for file, info in status["configuration"]["status"].items():
            if not info["exists"]:
                missing_components.append(file)
        
        for file, info in status["tests"]["status"].items():
            if not info["exists"]:
                missing_components.append(file)
        
        if missing_components:
            print("  1. Implement the following missing components:")
            for component in missing_components[:5]:  # Show at most 5 missing components
                print(f"     - {component}")
            if len(missing_components) > 5:
                print(f"     - ... and {len(missing_components) - 5} more")
        
        print("  2. Complete the dashboard implementation")
        print("  3. Add more comprehensive tests")
        print("  4. Push the code to GitHub")
    else:
        print("  All components have been implemented!")
        print("  Consider adding more tests and documentation.")

# Run the summary
print_project_summary()