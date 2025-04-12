import glob
import importlib.resources as resources
import os
import shutil

from cgcnn2.util import id_prop_gen


def atom_gen():
    """
    Copies the 'atom_init.json' asset from the cgcnn2 package's asset folder to
    the current working directory.

    The file is expected to be located at:
      cgcnn2/asset/atom_init.json
    within the installed package.
    """

    if os.path.exists("atom_init.json"):
        answer = (
            input(
                "atom_init.json already exists in the current directory. Overwrite? (y/n): "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            return

    try:
        with resources.path("cgcnn2.asset", "atom_init.json") as src_path:
            dest_path = os.path.join(os.getcwd(), "atom_init.json")
            shutil.copy(src_path, dest_path)
    except Exception as e:
        raise RuntimeError(f"Failed to copy atom_init.json: {e}")


def id_gen():
    """
    Generates an 'id_prop.csv' file in the current working directory.

    This function creates a CSV file with two columns: 'id' (derived from CIF filenames)
    and 'prop' (set to 0 for all entries).
    """

    if os.path.exists("id_prop.csv"):
        answer = (
            input(
                "id_prop.csv already exists in the current directory. Overwrite? (y/n): "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            return

    if not glob.glob("*.cif"):
        raise FileNotFoundError("No CIF files found in the current directory.")

    dest_path = os.path.join(os.getcwd())
    id_prop_gen(dest_path)
