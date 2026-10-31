import pytest

from charis.calibration_data import REGISTRY
from charis.instruments.instruments import instrument_from_data


def test_charis_header_without_filter_raises_without_interactive():
    header = {'INSTRUME': 'CHARIS'}
    with pytest.raises(ValueError, match="Y_FLTNAM"):
        instrument_from_data(header, calibration=True, interactive=False)


def test_charis_header_without_object_raises_without_interactive():
    header = {'INSTRUME': 'CHARIS', 'Y_FLTNAM': 'J'}
    with pytest.raises(ValueError, match="OBJECT"):
        instrument_from_data(header, calibration=False, interactive=False)


def test_charis_calibration_with_invalid_wavelength_raises(calibrations):
    header = {'INSTRUME': 'CHARIS', 'Y_FLTNAM': 'J', 'OBJECT': 'foo'}
    with pytest.raises(ValueError, match="1200.*1550.*2346"):
        instrument_from_data(header, calibration=True, interactive=False)


def test_charis_interactive_entry_keeps_static_calibdir(calibrations, monkeypatch):
    static_calibdir = str(calibrations / REGISTRY['CHARIS']['version'])
    answers = iter(["J", "1200"])
    monkeypatch.setattr("charis.instruments.instruments.input", lambda prompt: next(answers))
    instrument, wavelength, correct_header = instrument_from_data(
        {'INSTRUME': 'CHARIS'}, calibration=True, interactive=True, static_calibdir=static_calibdir)
    assert not correct_header
    assert instrument.calibration_version is None


def test_charis_valid_header_returns_instrument(calibrations):
    header = {'INSTRUME': 'CHARIS', 'Y_FLTNAM': 'J', 'OBJECT': '1200nm'}
    instrument, wavelength, correct_header = instrument_from_data(header, calibration=True)
    assert correct_header
    assert wavelength.value.tolist() == [1200]
