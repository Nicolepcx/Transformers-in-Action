import subprocess


def install_base_packages():
    packages = ["transformers==4.26.1", "datasets==2.10.1"]
    check = u'\u2705'
    print("\033[1mInstalling base requirements...\n\033[0m")
    for package in packages:
        process_scatter = subprocess.run(
            ["pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode != 0:
            print(f"Installation of {package} failed with error:\n{process_scatter.stderr.decode('utf-8')}")
        else:
            print(f"{check} {package} installation completed successfully!\n")


def install_required_packages_ch03():
    packages = ["summa==1.2.0", "evaluate==0.4.0", "rouge_score==0.1.2", "pyarrow==9.0.0", "sentencepiece"]
    check = u'\u2705'
    print("\033[1mInstalling chapter 3 requirements...\n\033[0m")
    for package in packages:
        process_scatter = subprocess.run(
            ["pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode != 0:
            print(f"Installation of {package} failed with error:\n{process_scatter.stderr.decode('utf-8')}")
        else:
            print(f"{check} {package} installation completed successfully!\n")

def install_required_packages_ch04():
    packages = ["transformers == 4.26.1", "datasets == 2.10.1", "evaluate==0.4.0", "rouge_score==0.1.2", "sacrebleu", "pyarrow==9.0.0", "sentencepiece"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 4 requirements...\n\033[0m")
    for package in packages:
        process_scatter = subprocess.run(
            ["pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode != 0:
            print(f"Installation of {package} failed with error:\n{process_scatter.stderr.decode('utf-8')}")
        else:
            print(f"{check} {package} installation completed successfully!\n")
