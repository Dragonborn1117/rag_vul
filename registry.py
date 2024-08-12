from __future__ import annotations
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from roles.__base import Role

def __init():
    global _global_dict
    _global_dict = {
        "Roles": {}
    }

def check_global(func):
    def wrapper(*args, **kwargs):
        try:
            _global_dict.get("Roles")
        except:
            __init()
        return func(*args, **kwargs)
    return wrapper

@check_global
def register_role(role: Type[Role]):
    _global_dict["Roles"][role.__name__] = role
    return role

@check_global
def get_registry():
    if not _global_dict:
        __init()
    return _global_dict

