import pkg_resources
import subprocess
import sys

# List of packages from your requirements.txt (without version constraints)
required_packages = [
    'torch',
    'torchvision',
    'numpy',
    'scipy',
    'h5py',
    'matplotlib',
    'tqdm',
    'pyyaml',
    'scikit-image',
    'tensorboard',
    'pytest',
    'pytest-cov',
    'fastmri',
    'einops',
    'wandb',
    'Pillow'
]

def get_installed_packages():
    """Get a set of installed package names."""
    installed = {pkg.key for pkg in pkg_resources.working_set}
    return installed

def install_package(package):
    """Install a package using pip in the current environment."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

def main():
    # Get installed packages
    installed_packages = get_installed_packages()
    print(f"Installed packages: {installed_packages}")

    # Check for missing packages
    missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
    
    if not missing_packages:
        print("All required packages are already installed in the FastReg environment.")
        return
    
    print(f"Missing packages: {missing_packages}")
    
    # Install missing packages
    for package in missing_packages:
        print(f"Installing {package}...")
        install_package(package)

if __name__ == "__main__":
    main()