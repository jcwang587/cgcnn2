import importlib.resources as resources
import os
import shutil


def atom_init():
    """
    Copies the 'atom_init.json' asset from the cgcnn2 package's asset folder to
    the current working directory.

    The file is expected to be located at:
      cgcnn2/asset/atom_init.json
    within the installed package.
    """
    try:
        with resources.path("cgcnn2.asset", "atom_init.json") as src_path:
            dest_path = os.path.join(os.getcwd(), "atom_init.json")
            shutil.copy(src_path, dest_path)
            print(f"'atom_init.json' successfully copied to {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred while copying 'atom_init.json': {e}")
