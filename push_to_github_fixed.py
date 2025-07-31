
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: push_to_github_fixed.py
# execution: true
import os
import subprocess

def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def setup_git_repo(repo_url, token, branch="main"):
    """Set up the Git repository."""
    # Check if git is installed
    if run_command("git --version") is None:
        print("Git is not installed or not in PATH.")
        return False
    
    # Initialize git repo if not already initialized
    if not os.path.exists(".git"):
        print("Initializing Git repository...")
        if run_command("git init") is None:
            return False
    
    # Configure Git credentials
    print("Configuring Git credentials...")
    run_command("git config --local user.name 'PromptShield Bot'")
    run_command("git config --local user.email 'bot@promptshield.ai'")
    
    # Add the remote repository with token authentication
    print(f"Adding remote repository: {repo_url}")
    # First remove any existing remote with the same name
    run_command("git remote remove origin 2>/dev/null || true")
    
    # Add the new remote with token authentication
    auth_url = repo_url.replace("https://", f"https://{token}@")
    if run_command(f"git remote add origin {auth_url}") is None:
        return False
    
    return True

def commit_and_push(branch="main"):
    """Commit all changes and push to the repository."""
    # Add all files
    print("Adding files to Git...")
    if run_command("git add .") is None:
        return False
    
    # Commit changes
    print("Committing changes...")
    commit_message = "Initial commit of PromptShield middleware"
    if run_command(f"git commit -m '{commit_message}'") is None:
        # If commit fails, it might be because there are no changes
        print("No changes to commit or commit failed.")
        return False
    
    # Create branch if it doesn't exist
    print(f"Creating branch: {branch}...")
    run_command(f"git checkout -b {branch}")
    
    # Push to the remote repository
    print(f"Pushing to branch: {branch}...")
    if run_command(f"git push -u origin {branch}") is None:
        return False
    
    return True

def list_files():
    """List all files that will be pushed."""
    files = []
    for root, dirs, filenames in os.walk("."):
        # Skip .git directory
        if ".git" in dirs:
            dirs.remove(".git")
        
        # Skip any hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for filename in filenames:
            # Skip hidden files and temporary files
            if not filename.startswith(".") and not filename.startswith("tmp_"):
                path = os.path.join(root, filename)
                files.append(path)
    
    return files

def main():
    """Main function to push code to GitHub."""
    # Repository URL and token from the requirements
    repo_url = "https://github.com/harshadindigal/PromptGuard"
    token = "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"  # Token from the original requirements
    branch = "main"
    
    print("Preparing to push files to GitHub...")
    print(f"Repository: {repo_url}")
    print(f"Branch: {branch}")
    
    # List files that will be pushed
    files = list_files()
    print(f"\nFound {len(files)} files to push:")
    for file in files[:10]:  # Show first 10 files
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    # Set up the repository
    if not setup_git_repo(repo_url, token, branch):
        print("Failed to set up Git repository.")
        return
    
    # Commit and push changes
    if commit_and_push(branch):
        print(f"\nSuccessfully pushed code to {repo_url} on branch {branch}.")
    else:
        print("\nFailed to push code to GitHub.")
        print("\nThis might be because the token has expired or lacks permissions.")
        print("To push the code manually, you can use these commands:")
        print("\n  git checkout -b main")
        print("  git push -u origin main")
        print("\nOr if you need to authenticate with a new token:")
        print("  git remote set-url origin https://USERNAME:NEW_TOKEN@github.com/harshadindigal/PromptGuard.git")
        print("  git push -u origin main")

if __name__ == "__main__":
    main()