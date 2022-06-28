import cmd
import csv
import datetime
import os
import string
from functools import reduce
from typing import Optional
import pandas as pd
import yaml
from yaml.loader import SafeLoader


def print_error(e: Exception):
    try:
        print(
            f"\n\033[91m[{e.__class__.__name__}] {e.strerror}: '{e.filename}'\033[00m"
        )
    except AttributeError:
        print(f"\n\033[91m[{e.__class__.__name__}] {e.args[0]}\033[00m")


class ConfigShell(cmd.Cmd):
    # skip validation commands
    skip_validation = ["read", "clear", "help", "exit"]
    prompt = "(Configuration)>>>"
    ruler = ""
    intro = "Start Configuration Shell"

    def __init__(self, file: str = "", preset: Optional[str] = None) -> None:
        """Initialize the ConfigShell class.

        Args:
            file (str): path to the configuration file
        Errors:
            FileNotFoundError: The file is invalid or does not exist.
            pandas.errors.EmptyDataError: The file is empty.
            pandas.errors.ParserError: Parameter keys must be in row 2(header=1) of the file.
            IndexError: There is no recent log in the file.
        """
        super().__init__()

        # [SET display setting]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)

        # [SET configuration parameter]
        self._write_params = True
        self._add_params = False
        # self.use_default = False
        self.parameter = dict()
        self.file = None
        if preset:
            self.do_read(preset)
        else:
            self.do_read(file)
        self.cmdloop()

    def precmd(self, line: str) -> str:
        if any(line.startswith(command) for command in self.skip_validation):
            pass
        else:
            # validate the configuration file exists
            if not self.file:
                print(
                    "Please read the configuration file first.\n"
                    "If the file is in the current directory, please use `read ./filename`\n"
                )
                return ""
        return super().precmd(line)

    @staticmethod
    def emptyline() -> None:
        return

    def do_read(self, file: str) -> None:
        """Read the configuration file.

        Prompt:
            (Configuration)>>>read [path]

        Errors:
            FileNotFoundError: The file is invalid or does not exist.
            pandas.errors.EmptyDataError: The file is empty.
            pandas.errors.ParserError: Parameter keys must be in row 2(header=1) of the file.
            IndexError: There is no recent log in the file.
        """
        if file:
            file, *_ = file.split()
            filename, ext = os.path.splitext(file)
            try:
                if ext == ".csv":
                    self.parameter = pd.read_csv(file, header=None)
                    self.parameter.iloc[0].fillna(method="ffill", inplace=True)
                elif ext in [".yml", ".yaml"]:
                    with open(file, "r") as f:
                        self.parameter = yaml.load(f, Loader=SafeLoader)
                else:
                    raise FileNotFoundError
            except FileNotFoundError as e:
                # FileNotFoundError: The file is invalid or does not exist.
                print_error(e)
                print(
                    "The file is invalid or does not exist. Please check the file path.\n"
                    "If the file is in the current directory, please use `read ./filename`\n"
                )
            except pd.errors.EmptyDataError as e:
                # pandas.errors.EmptyDataError: The file is empty.
                print_error(e)
                print("The file is empty.")
            except pd.errors.ParserError as e:
                # pd.errors.ParserError: Parameter keys must be in row 2(header=1) of the file.
                print_error(e)
                print(
                    "Parameter keys must be in row 2(header=1) of the file.\n"
                    "Please check the file format.\n"
                )
            except IndexError as e:
                # IndexError: There is no recent log in the file.
                print_error(e)
                print("There is no recent log in the file.")
            except Exception as e:
                print_error(e)
                print("Unknown error.")
            else:
                self.file = file
                print(f"Config file : {os.path.abspath(self.file)}")
        else:
            print("Please input the file path.")

    @staticmethod
    def complete_path(curdir, item) -> str:
        return item if os.path.isfile(curdir + item) else item + os.path.sep

    @staticmethod
    def ignore(item) -> bool:
        return not any(item.startswith(p) for p in string.punctuation)

    def complete_read(self, text, line, begidx, endidx) -> list:
        # get current directory
        try:
            _, curdir = line[:begidx].split()
        except:
            curdir = "./"
        curdir = os.path.expanduser(curdir)

        # get list of files in current directory
        items = os.listdir(curdir)
        filtered_items = filter(self.ignore, items)

        return [
            self.complete_path(curdir, item)
            for item in filtered_items
            if (item.startswith(text) if text else True)
        ]

    def do_show(self, params=None) -> None:
        """Show the configuration parameters.

        Prompt:
            (Configuration)>>>show
            (Configuration)>>>show [parameter]
        """
        filtered_params = self.parameter
        if params:
            filtered_params = {
                param: value
                for param in params.split()
                if (value := reduce(dict.get, param.split("."), self.parameter))
            }
        if filtered_params:
            print(yaml.dump(filtered_params, sort_keys=False))

    def do_set(self, kwargs: str) -> None:
        """Set the configuration parameters.

        Prompt:
            (Configuration)>>>set [key] [value]
        """
        try:
            key, *args = kwargs.split()  # ValueError
            *keys, last_key = key.split(".")
            deepest_dict = reduce(dict.setdefault, keys, self.parameter)
            prev = deepest_dict[last_key]  # KeyError
            if isinstance(prev, dict):
                raise Exception(f"{last_key} cannot be overwritten directly.")
            value = type(prev)(args[0]) if args[0] != "None" else None  # ValueError
        except Exception as e:
            print_error(e)
        else:
            deepest_dict.update({last_key: value})
            print(key, f": {type(value).__name__} = {prev} -> {value}")
            if len(args) > 2:
                del args[0]
                self.do_set(" ".join(args))

    def completedefault(self, text, line, begidx, endidx) -> list:
        params = frozenset(
            pd.json_normalize(self.parameter).to_dict(orient="records")[0]
        )
        if text:
            return [param for param in params if param.startswith(text)]
        else:
            return list(params)
        """ try:
            *query, _ = text.split(".")
            params = reduce(dict.get, query, self.parameter).keys()
        except:
            params = self.parameter.keys()

        if not text.endswith("."):
            *prefix, query = text.split(".")
            return [
                ".".join(prefix + [param])
                for param in params
                if param.startswith(query)
            ]
        elif len(params) == 1:
            return [text + list(params)[0]]
        else:
            return list(params)"""

    @staticmethod
    def do_clear(_) -> None:
        """Clear the screen.

        Prompt:
            (Configuration)>>>clear
        """
        os.system("clear")

    @staticmethod
    def do_done(_) -> None:
        """Exit the configuration shell and Start the main script.

        Prompt:
            (Configuration)>>>done
        Shortcut:
            Ctrl+D
        """
        print("The configuration is all set.")
        return True

    do_EOF = do_done

    @staticmethod
    def do_exit(_) -> None:
        """Exit the configuration shell and Exit the program.

        Prompt:
            (Configuration)>>>exit
        """
        raise SystemExit("Configuration failed. Abort Experiment")

    def write_parameters(self):
        """Write the configuration file."""
        with open(self.file, "a+", newline="") as file:
            csv_dict_writer = csv.DictWriter(
                file, fieldnames=list(self.parameter.keys())
            )
            csv_dict_writer.writerow(self.parameter)


if __name__ == "__main__":
    configsh = ConfigShell("config/preset/example.yaml")
