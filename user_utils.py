# Define useful functions to interact with the user
from typing import Union
from colorama import just_fix_windows_console
just_fix_windows_console()

# Create a class called FontColorDummy to store a string but not being an instance of str
class FontColorDummy:
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


# Print text with color
class FontColor:
    PURPLE = FontColorDummy("\033[95m")
    CYAN = FontColorDummy("\033[96m")
    DARKCYAN = FontColorDummy("\033[36m")
    BLUE = FontColorDummy("\033[94m")
    GREEN = FontColorDummy("\033[92m")
    YELLOW = FontColorDummy("\033[93m")
    RED = FontColorDummy("\033[91m")
    BOLD = FontColorDummy("\033[1m")
    UNDERLINE = FontColorDummy("\033[4m")
    END = FontColorDummy("\033[0m")


def str_to_font_color(color: str) -> FontColor:
    """Convert a string to a FontColor enum."""
    color = color.upper().replace(" ", "").replace("-", "").replace("_", "")
    if color == "PURPLE":
        return FontColor.PURPLE
    elif color == "CYAN":
        return FontColor.CYAN
    elif color == "DARKCYAN":
        return FontColor.DARKCYAN
    elif color == "BLUE":
        return FontColor.BLUE
    elif color == "GREEN":
        return FontColor.GREEN
    elif color == "YELLOW":
        return FontColor.YELLOW
    elif color == "RED":
        return FontColor.RED
    elif color == "BOLD":
        return FontColor.BOLD
    elif color == "UNDERLINE":
        return FontColor.UNDERLINE
    else:
        raise ValueError("Unknown color: {}".format(color))


def get_with_color(text: str, color: Union[FontColorDummy, str, list]) -> str:
    """Get text with color(s)"""
    if isinstance(color, str):
        color = str_to_font_color(color)
    if isinstance(color, list):
        if len(color) == 0:
            return text
        elif len(color) == 1:
            return get_with_color(text, color[0])
        return get_with_color(get_with_color(text, color[0]), color[1:])
    return str(color) + text + str(FontColor.END)


def ask_yes_no_question(question: str, color: Union[str, list] = None) -> bool:
    """Ask a yes/no question to the user."""
    if color is None:
        color = FontColor.BOLD
    question = get_with_color(question+ " (y/n): ", color)
    answer = None
    while answer not in ["y", "n"]:
        answer = input(question).lower()
        if answer not in ["y", "n"]:
            print(get_with_color("Please answer with 'y' or 'n'.", FontColor.RED))
    return answer == "y"


def warn(text: str, severity: int = 1):
    """Print a warning message."""
    if severity == 1:
        print(get_with_color("Warning: ", FontColor.YELLOW) + text)
    elif severity == 2:
        print(get_with_color("Warning: ", FontColor.RED) + text)
    else:
        print(get_with_color("Warning: ", [FontColor.BOLD, FontColor.RED]) + text)


if __name__ == "__main__":
    warn("This is a warning", 1)
    ask_yes_no_question("Do you want to continue?", [FontColor.BOLD, "purple"])