"""
Stylising the Program

"""
from tkinter import font


def headerStyles(family: str, base_font_size: int):
    """
    Header Styles
    """
    cust_text = font.Font(family=family, size=base_font_size)
    cust_subheader = font.Font(family=family, size=(base_font_size + 2), weight='bold')
    cust_subheadernb = font.Font(family=family, size=(base_font_size + 2))
    cust_header = font.Font(family=family, size=(base_font_size + 4))
    return cust_text, cust_subheader, cust_subheadernb, cust_header
