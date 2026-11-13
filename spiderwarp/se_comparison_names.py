"""Human-readable comparison names; saved protocol IDs and hashes stay stable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonMethod:
    name: str
    qubits: int
    results: str

    @property
    def label(self):
        return f'{self.name} ({self.qubits} qubits)'


METHODS = {
    'adaptive_baseline': ComparisonMethod('Adaptive baseline', 60, 'se_100k'),
    'adaptive_history': ComparisonMethod('Adaptive history PS', 60, 'se_postselected_100k'),
    'adaptive_syndrome': ComparisonMethod('Adaptive syndrome PS', 60, 'adaptive_css_100k'),
    'adaptive_recovery': ComparisonMethod('Adaptive verified recovery', 60, 'adaptive_css_recover_safe_100k'),
    'steane_d4': ComparisonMethod('Steane d4 RUS', 66, 'steane_rus_d4_100k'),
    'steane_descending': ComparisonMethod('Steane descending RUS', 66, 'steane_rus_ladder_ancilla_100k'),
    'steane_d4_ps': ComparisonMethod('Steane d4 RUS + syndrome PS', 66, 'steane_css_ps_d4_100k'),
    'steane_descending_ps': ComparisonMethod('Steane descending RUS + syndrome PS', 66, 'steane_css_ps_ladder_ancilla_100k'),
}

PILOT_NAMES = {
    'css': 'Syndrome PS',
    'css-flag': 'Syndrome PS + flagged-history check',
    'css-flag-reversed': 'Syndrome PS + X/Z reversal',
    'css-flag-check': 'Syndrome PS + final d2 checks',
    'css-separated': 'Separate CSS histories',
    'css-separated-retarget': 'Separate CSS histories + frame update',
    'css-recover': 'Bounded recovery',
    'css-recover-exact': 'Exact recovery without the EC filter constraint',
    'css-recover-no-d1': 'Bounded recovery without d1',
    'css-recover-safe': 'Verified recovery',
}
