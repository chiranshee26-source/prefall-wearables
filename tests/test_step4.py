import json
import re
from prefall.alert import pack_alert, unpack_alert, SIZE
from prefall.demo import build_html


def test_payload_is_8_bytes_and_round_trips():
    b = pack_alert(123456, 0x0A01, 0.97, policy=1)
    assert len(b) == SIZE == 8
    d = unpack_alert(b)
    assert d["t_ms"] == 123456 and d["device_id"] == 0x0A01 and d["policy"] == 1
    assert abs(d["confidence"] - 0.97) < 1 / 255


def test_payload_clips_and_wraps():
    assert unpack_alert(pack_alert(0, 1, 1.7))["confidence"] == 1.0
    assert unpack_alert(pack_alert(0, 1, -3))["confidence"] == 0.0
    assert unpack_alert(pack_alert(2 ** 32 + 5, 1, 0.5))["t_ms"] == 5


def test_demo_page_embeds_data():
    data = dict(scenarios=[dict(label="x")], params=dict(theta_crit=15))
    html = build_html(data)
    assert "__DATA__" not in html
    m = re.search(r"const D = (\{.*?\});\nconst \$", html, re.S)
    assert json.loads(m.group(1)) == data
