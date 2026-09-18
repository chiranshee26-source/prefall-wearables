"""Builds the self-contained caregiver demo page from precomputed scenario data."""
import json
import os

TEMPLATE = os.path.join(os.path.dirname(__file__), "demo_template.html")


def build_html(data):
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    return html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
