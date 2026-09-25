from pathlib import Path

import numpy as np
import pytest

from dispick.physics.bank import BankConfig, ModalBank, build_bank
from dispick.synthesis.generator import SyntheticGenerator


@pytest.fixture(scope="session")
def bank_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny bank, built once for the whole session."""
    path = tmp_path_factory.mktemp("bank") / "bank.h5"
    return build_bank(BankConfig(n_models=24, seed=3), path, workers=2, chunk=4)


@pytest.fixture(scope="session")
def bank(bank_path: Path) -> ModalBank:
    return ModalBank(bank_path, in_memory=True)


@pytest.fixture(scope="session")
def generator(bank: ModalBank) -> SyntheticGenerator:
    return SyntheticGenerator(bank)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
