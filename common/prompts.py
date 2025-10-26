from __future__ import annotations
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

def get_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )

def render_template(template_dir: str, template_name: str, **kwargs) -> str:
    env = get_env(template_dir)
    tpl = env.get_template(template_name)
    return tpl.render(**kwargs)
