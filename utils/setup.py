import torch
def useGPU():
    is_gpu_available = torch.cuda.is_available()
    snail = u'\U0001F40C'
    rocket = u'\U0001F680'
    check = u'\u2705'
    party_emoji  = u'\U0001F973'

    if not is_gpu_available:
        print(f"No GPU detected! This notebook will be\033[1;35m\033[5m very slow {snail}. \033[0m")
        print(f"Consider uploading the notebook to one of the cloud platforms \033[1;31m\033[5m for much faster computation\033[0m {rocket}.\n")
        print(f"{check} \u001b[32mAlso, remember you might have to change the runtime type of your notebook!\032")
    else:
        print(f"Have fun with this chapter!{party_emoji}")
