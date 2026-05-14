# =========================================================
# SYSTEM INFORMATION LOGGER
# =========================================================

import platform

import psutil


# =========================================================
# GET SYSTEM INFO
# =========================================================

def get_system_info():

    system_info = {

        "Operating System":
        platform.system(),

        "OS Version":
        platform.version(),

        "Python Version":
        platform.python_version(),

        "Processor":
        platform.processor(),

        "RAM (GB)":
        round(

            psutil.virtual_memory().total

            / (1024 ** 3),

            2
        )
    }

    return system_info