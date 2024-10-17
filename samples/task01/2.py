from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    paren_string = paren_string.replace(" ", "")
    parenthesis_list = []
    for i in range(len(paren_string)):
        if paren_string[i] == "(":
            group_count = 1
            subgroup_start_index = i
            for j in range(i + 1, len(paren_string)):
                if paren_string[j] == "(":
                    