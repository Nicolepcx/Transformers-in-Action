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

def install_required_packages_ch05():
    packages = ["transformers == 4.26.1", "datasets == 2.10.1", "evaluate==0.4.0", "pyarrow==9.0.0", "sentencepiece"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 5 requirements...\n\033[0m")
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

def install_required_packages_ch06():
    packages = ["accelerate==0.26.1"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 6 requirements...\n\033[0m")
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

def install_required_packages_ch07():
    packages = ["accelerate==0.26.1", "wandb", "peft==0.7.1", "safetensors==0.4.1", "trl==0.7.10", "tree-of-thoughts-llm==0.1.0"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 7 requirements...\n\033[0m")
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

def install_required_packages_ch08():
    packages = ["accelerate==0.26.1", "peft==0.7.1", "safetensors==0.4.1", "bitsandbytes==0.43.0", "gradio==4.21.0", "transformers == 4.38.2"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 8 requirements...\n\033[0m")
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

def install_required_packages_ch09():
    packages = ["accelerate==0.26.1", "safetensors==0.4.1",
                 "transformers == 4.38.2", "datasets==2.10.1",
                "torch>=1.10.0", "ray==2.9.3", "wandb", "bitsandbytes==0.43.0"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 9 requirements...\n\033[0m")
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


def install_required_packages_ch10():
    packages = ["accelerate==0.26.1", "safetensors==0.4.1", "captum==0.7.0",
                 "transformers == 4.38.2", "bitsandbytes==0.43.0", "llm-guard==0.3.10"]

    check = u'\u2705'
    print("\033[1mInstalling chapter 10 requirements...\n\033[0m")
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
