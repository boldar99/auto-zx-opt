"""Check the selected rescue graph against the parent's order-one EC contract."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from spiderwarp._se_adaptive import FTContract, FixedCSSDecoder, AdaptiveVerificationReport
from spiderwarp.adaptive_steane_se import AdaptiveSEProtocol
from spiderwarp.se_protocol_io import load_protocol, read_manifest


if __name__ == '__main__':
    path = Path('build/adaptive_css_recover_exact_32_20_4')
    parent = load_protocol(path)
    contract = FTContract(1, 1, fault_order=1, accept_clean_through=1)
    decoder = FixedCSSDecoder(parent.policy.decoder.space, contract)
    policy = replace(parent.policy, contract=contract, decoder=decoder,
                     report=AdaptiveVerificationReport(contract, decoder.fingerprint))
    trial = AdaptiveSEProtocol(parent.code, policy)
    result = dict(protocol_hash=read_manifest(path)[1]['protocol_hash'],
                  graph_and_quantum_actions_unchanged=True, tested_contract=asdict(contract))
    try:
        trial.verify()
    except Exception as error:
        result.update(passed=False, error_type=type(error).__name__, reason=str(error))
    else:
        result.update(passed=True, report=asdict(trial.verification))
    target = Path(__file__).resolve().parent / 'stronger_contract.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
