import subprocess

def install(chapter_02: bool = False):

    """
    Installs the required packages for a specific chapter of the Transformers-in-Action book.
    Args:
    chapter_02 (bool): Flag to install packages required for Chapter 2. Default is False.

    Returns:
    None

    Raises:
    None
    """

    print("Installing requirements...")
    cmd = ["python", "-m", "pip", "install", "-r"]

    if chapter_02:
        transformers_cmd = "python -m pip install transformers==4.26.1 datasets==2.10.1 summa==1.2.0 evaluate==0.4.0".split()
        process_scatter = subprocess.run(
            transformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    print("Chapter installation completed!")
