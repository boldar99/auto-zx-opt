from types import SimpleNamespace

import stim

from spiderwarp.se_protocol_inspection import (
    circuit_resources, circuit_inspection_html, protocol_circuit_roles,
)


def test_resource_counts_use_lowered_rotations_and_exclude_dead_idle_time():
    circuit = stim.Circuit('RX 1\nM 1\nH 0\nH 0\nH 0\nH 0\nH 0')
    before = circuit.copy()
    resources = circuit_resources(circuit, 1)
    assert resources['preparations'] == resources['measurements'] == 1
    assert resources['one_qubit_gates'] == 6
    assert resources['idle_units'] > resources['live_idle_units']
    assert circuit == before


def test_notebook_view_keeps_native_svg_size_and_full_parities():
    circuit = stim.Circuit('R 1\nCX 0 1\nM 1\nDETECTOR(0, 0) rec[-1]')
    html = circuit_inspection_html(circuit, '<X> extraction', 1)
    assert '&lt;X&gt; extraction' in html
    assert 'max-width:none' in html and 'overflow:auto' in html
    assert 'DETECTOR(0, 0) rec[-1]' in html
    assert '<details>' in html


def test_inventory_distinguishes_selected_and_unused_circuits():
    protocol = SimpleNamespace(
        code=SimpleNamespace(d=4),
        gadgets={('X', d): None for d in (1, 2, 4)} | {('Z', d): None for d in (1, 2, 4)},
        recoveries={0: SimpleNamespace(primary_basis='Z', distance=1, cleanup_distance=2)},
    )
    roles = protocol_circuit_roles(protocol)
    assert roles['Z', 4] == roles['X', 4] == ('primary',)
    assert roles['X', 1] == ('recovery',)
    assert roles['Z', 2] == ('cleanup',)
    assert roles['X', 2] == roles['Z', 1] == ()
