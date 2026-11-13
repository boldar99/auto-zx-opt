"""Human-readable comparison names; saved protocol IDs and hashes stay stable."""
from dataclasses import dataclass


PS_RULES = {
    'history': ('r+s≤1', r'r+s\leq1'),
    'syndrome': ('r_syn≤1', r'r_{\mathrm{syn}}\leq1'),
    'history_syndrome': ('r+s≤1, r_syn≤1', r'r+s\leq1,\ r_{\mathrm{syn}}\leq1'),
    'no_flags_syndrome': ('f=0, r_syn≤1', r'f=0,\ r_{\mathrm{syn}}\leq1'),
    'recovery': ('r+s≤1 or V', r'r+s\leq1\ \mathrm{or}\ V'),
    'order_two': ('r+s≤1 or V₂', r'r+s\leq1\ \mathrm{or}\ V_2'),
    'probability_10': ('r+s≤1 or p_err≤10%', r'r+s\leq1\ \mathrm{or}\ p_{\rm err}\leq10\%'),
    'probability_25': ('r+s≤1 or p_err≤25%', r'r+s\leq1\ \mathrm{or}\ p_{\rm err}\leq25\%'),
}
LEGEND_NOTE = (r'$r+s\leq1$: compatible history; $V_2$: exact uniqueness; '
               r'$p_{\rm err}$: estimated MAP correction error; $f=0$: zero flags; '
               r'$r_{\rm syn}$: CSS order. LER/cost exclude rejected shots.')


@dataclass(frozen=True)
class ComparisonMethod:
    title: str
    qubits: int
    results: str
    family: str = 'adaptive'
    postselection: str | None = None

    @property
    def name(self):
        return self.title + (f' PS({PS_RULES[self.postselection][0]})' if self.postselection else '')

    @property
    def label(self):
        return f'{self.name} ({self.qubits} qubits)'

    @property
    def plot_label(self):
        title = self.title + (f' PS(${PS_RULES[self.postselection][1]}$)' if self.postselection else '')
        return f'{title} ({self.qubits} qubits)'

    @property
    def marker(self):
        return 's' if self.family == 'steane' else 'o'


METHODS = {
    'adaptive_history': ComparisonMethod('Adaptive', 60, 'se_postselected_100k', postselection='history'),
    'adaptive_order_two': ComparisonMethod('Adaptive', 60, 'adaptive_order_two_100k', postselection='order_two'),
    'adaptive_probability_10': ComparisonMethod('Adaptive', 60, 'adaptive_probability_10_100k', postselection='probability_10'),
    'adaptive_probability_25': ComparisonMethod('Adaptive', 60, 'adaptive_probability_25_100k', postselection='probability_25'),
    'adaptive_no_flags': ComparisonMethod('Adaptive', 60, 'adaptive_css_no_flags_100k', postselection='no_flags_syndrome'),
    'steane_fixed_ps': ComparisonMethod('Steane d3 RUS', 66, 'steane_single_d3_fixed_ps_100k_accepted', family='steane', postselection='syndrome'),
}

METHOD_COLORS = {
    'adaptive_history': '#d62728',
    'adaptive_order_two': '#1f77b4',
    'adaptive_probability_10': '#2ca02c',
    'adaptive_probability_25': '#8c564b',
    'adaptive_no_flags': '#ff7f0e',
    'steane_fixed_ps': '#9467bd',
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

# These fixed-budget discovery/holdout pilots are intentionally separate from
# METHODS: the canonical comparison still requires full matched production runs.
LIKELIHOOD_PILOT_NAMES = {
    'likelihood-1': 'Adaptive likelihood PS(ε=1%)',
    'likelihood-5': 'Adaptive likelihood PS(ε=5%)',
    'likelihood-25': 'Adaptive likelihood PS(ε=25%)',
}
