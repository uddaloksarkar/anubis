from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    paren_groups: List[str] = []
    if not paren_string:
        return paren_groups
    open_count = 0
    paren_group = ""
    for paren in paren_string:
        if paren == "(":
            open_count += 1
        elif paren == ")":
            open_count -= 1
        elif paren == " " and open_count == 0:
