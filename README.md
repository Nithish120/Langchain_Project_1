Steps:
->First create a New Folder

->Install UV in terminal. UV is a python package manager similar to pip. Using UV is very useful as it automates many processes like creating virtual environments and basic boiler plate codes.

->First use the command uv init. This created basic main.py files then it created .toml files which contains dependencies.

->Next use uv add command to add any dependencies or libraries.

->When uv add is used it automatically creates virtual environments.

->Then we must add python dot env library because it is used to store API keys in .env file. API keys must be stored in .env files because when applications are running they may directly take the API keys from .env files.

->Then python .gitignore file must be created and the standard code is pasted in it


