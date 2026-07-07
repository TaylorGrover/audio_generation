import os
def getOS():
    if os.environ.get("OS", "Linux") == "Windows_NT":
        return "windows"
    else:
        return "linux"
