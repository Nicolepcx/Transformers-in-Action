import subprocess

def install(
    chapter_03=False,
):
    """
    Installs the required packages for a specific chapter of the Transformers-in-Action book.
    Args:
    chapter_03 (bool): Flag to install packages required for Chapter 3. Default is False.

    Returns:
    None

    Raises:
    None
    """
    check = u'\u2705'

    print("Installing requirements...\n")
    base_cmd = "python -m pip install transformers==4.26.1 datasets==2.10.1".split()
    process_scatter = subprocess.run(
        base_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    eval_cmd = "python -m pip install summa==1.2.0 evaluate==0.4.0".split()
    process_scatter = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if chapter_03:
        print("Installing requirements for chapter 3...")
        chapter_03_cmd = "python -m pip install rouge_score==1.4.0".split()
        process_scatter = subprocess.run(
            chapter_03_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Chapter 3 installation completed!\n")

    print(f"{check} All installations completed!")
